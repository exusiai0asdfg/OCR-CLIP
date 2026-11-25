"""Read selected image.bytes pages with Parquet's OffsetIndex."""

from bisect import bisect_right
import io
import struct

import thriftpy2
from thriftpy2.protocol import TCompactProtocolFactory
from thriftpy2.utils import deserialize, serialize


# Subset of the wire schema in apache/parquet-format's parquet.thrift.
# The original compressed pages are passed to PyArrow with a small local footer.
FORMAT = thriftpy2.load_fp(io.StringIO("""
struct SchemaElement {
    1: optional i32 type,
    3: optional i32 repetition_type,
    4: required string name,
    5: optional i32 num_children
}
struct ColumnMetadata {
    1: required i32 type,
    2: required list<i32> encodings,
    3: required list<string> path_in_schema,
    4: required i32 codec,
    5: required i64 num_values,
    6: required i64 total_uncompressed_size,
    7: required i64 total_compressed_size,
    9: required i64 data_page_offset,
    11: optional i64 dictionary_page_offset
}
struct ColumnChunk {
    2: required i64 file_offset,
    3: optional ColumnMetadata meta_data,
    4: optional i64 offset_index_offset,
    5: optional i32 offset_index_length
}
struct RowGroup {
    1: required list<ColumnChunk> columns,
    2: required i64 total_byte_size,
    3: required i64 num_rows
}
struct FileMetadata {
    1: required i32 version,
    2: required list<SchemaElement> schema,
    3: required i64 num_rows,
    4: required list<RowGroup> row_groups
}
struct PageLocation {
    1: required i64 offset,
    2: required i32 compressed_page_size,
    3: required i64 first_row_index
}
struct OffsetIndex {
    1: required list<PageLocation> page_locations
}
struct DataPage {
    1: required i32 num_values,
    2: required i32 encoding
}
struct DataPageV2 {
    1: required i32 num_values,
    4: required i32 encoding
}
struct PageHeader {
    1: required i32 type,
    2: required i32 uncompressed_page_size,
    3: required i32 compressed_page_size,
    5: optional DataPage data_page_header,
    8: optional DataPageV2 data_page_header_v2
}
"""), module_name="ocl_imagenet_page_thrift")
COMPACT = TCompactProtocolFactory()
MAGIC = b"PAR1"
BYTE_ARRAY = 6
OPTIONAL = 1
DICTIONARY_ENCODINGS = {2, 8}


def parse_thrift(cls, data):
    try:
        return deserialize(cls(), data, COMPACT)
    except (EOFError, ValueError, TypeError, struct.error) as error:
        raise ValueError("Cannot decode Parquet compact metadata.") from error


def read_at(file, offset, length):
    if offset < 4 or length <= 0:
        raise ValueError("Invalid Parquet range offset or length.")
    file.seek(offset)
    data = file.read(length)
    if len(data) != length:
        raise ValueError("Truncated Parquet range.")
    return data


def read_footer(file):
    end = file.seek(0, io.SEEK_END)
    length, magic = struct.unpack("<I4s", read_at(file, end - 8, 8))
    if magic != MAGIC or length > 2 * 1024 * 1024:
        raise ValueError("Invalid or oversized Parquet footer.")
    return parse_thrift(FORMAT.FileMetadata, read_at(file, end - 8 - length, length))


def page_header(packet):
    header = parse_thrift(FORMAT.PageHeader, packet)
    if not 0 < header.compressed_page_size < len(packet):
        raise ValueError("Parquet page length does not agree with its header.")
    return header


def decode_page(pq, metadata, schema, packet, header, num_rows, dictionary, dictionary_header):
    uncompressed = len(packet) - header.compressed_page_size + header.uncompressed_page_size
    if dictionary:
        uncompressed += len(dictionary) - dictionary_header.compressed_page_size + dictionary_header.uncompressed_page_size
    column = FORMAT.ColumnMetadata(
        type=metadata.type, encodings=metadata.encodings, path_in_schema=metadata.path_in_schema,
        codec=metadata.codec, num_values=num_rows, total_uncompressed_size=uncompressed,
        total_compressed_size=len(packet) + len(dictionary), data_page_offset=4 + len(dictionary),
        dictionary_page_offset=4 if dictionary else None,
    )
    row_group = FORMAT.RowGroup(
        columns=[FORMAT.ColumnChunk(file_offset=0, meta_data=column)],
        total_byte_size=uncompressed, num_rows=num_rows,
    )
    footer = serialize(FORMAT.FileMetadata(version=1, schema=schema, num_rows=num_rows, row_groups=[row_group]), COMPACT)
    buffer = io.BytesIO(b"".join((MAGIC, dictionary, packet, footer, struct.pack("<I", len(footer)), MAGIC)))
    with pq.ParquetFile(buffer, page_checksum_verification=True) as reader:
        return reader.read_row_group(0, use_threads=False)["image"].combine_chunks().field("bytes")


def read_image_bytes(pq, file, parquet, footer, group, row_indexes):
    index = next(i for i in range(len(parquet.schema)) if parquet.schema.column(i).path == "image.bytes")
    column = parquet.schema.column(index)
    chunk = footer.row_groups[group].columns[index]
    row_count = footer.row_groups[group].num_rows
    metadata = chunk.meta_data
    if (
        metadata.path_in_schema != ["image", "bytes"] or metadata.type != BYTE_ARRAY
        or column.max_repetition_level != 0 or column.max_definition_level != 2
    ):
        raise ValueError("Unexpected image.bytes schema for Parquet page reads.")
    if not chunk.offset_index_offset or not chunk.offset_index_length:
        raise ValueError("Image column has no OffsetIndex. A whole row group would be required.")
    raw_index = read_at(file, chunk.offset_index_offset, chunk.offset_index_length)
    pages = parse_thrift(FORMAT.OffsetIndex, raw_index).page_locations
    starts = [page.first_row_index for page in pages]
    if not starts or starts[0] != 0 or starts[-1] >= row_count or any(
        a >= b for a, b in zip(starts, starts[1:])
    ):
        raise ValueError("Parquet column page index has invalid row boundaries.")
    ends = starts[1:] + [row_count]
    page_matches = {}
    for row in row_indexes:
        if not 0 <= row < row_count:
            raise ValueError("Image record index is out of bounds.")
        page_matches.setdefault(bisect_right(starts, row) - 1, []).append(row)

    schema = [
        FORMAT.SchemaElement(name="schema", num_children=1),
        FORMAT.SchemaElement(name="image", num_children=1, repetition_type=OPTIONAL),
        FORMAT.SchemaElement(name="bytes", repetition_type=OPTIONAL, type=BYTE_ARRAY),
    ]
    results = {}
    dictionary, dictionary_header = b"", None
    for i, matched in page_matches.items():
        page = pages[i]
        packet = read_at(file, page.offset, page.compressed_page_size)
        header = page_header(packet)
        data_header = header.data_page_header or header.data_page_header_v2
        num_rows = ends[i] - starts[i]
        if data_header is None or data_header.num_values != num_rows:
            raise ValueError("Parquet page row count disagrees with OffsetIndex.")
        uses_dictionary = data_header.encoding in DICTIONARY_ENCODINGS
        if uses_dictionary and not dictionary:
            start = metadata.dictionary_page_offset
            length = metadata.data_page_offset - start if start else 0
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("Image page requires a missing or oversized Parquet dictionary.")
            dictionary = read_at(file, start, length)
            dictionary_header = page_header(dictionary)
        values = decode_page(
            pq, metadata, schema, packet, header, num_rows,
            dictionary if uses_dictionary else b"", dictionary_header,
        )
        results.update((row, values[row - starts[i]].as_py()) for row in matched)
    return results

"""Bounded, cached recovery of exact OCL Web assets."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, urlsplit, urlunsplit
import zlib

import requests

if __package__:
    from . import download_web as web
else:
    import download_web as web


RELEASE = "20231130235959"
CC = "https://data.commoncrawl.org/"
INDEX_PATH = "cc-index/collections/{crawl}/indexes/"
IMAGE_MIME = ("image/", "application/octet-stream", "binary/octet-stream")
CC_CRAWLS = {"CC-MAIN-2023-40":167732858, "CC-MAIN-2020-50":136024962, "CC-MAIN-2018-51":149109429}
DEFAULT_CRAWLS = ["CC-MAIN-2023-40","CC-MAIN-2020-50"]
TRANSIENT = {"network_failure", "api_unavailable"}


class Failure(Exception):
    def __init__(self, category, detail):
        super().__init__(detail)
        self.category = category
        self.detail = detail


class Service:
    def __init__(self, interval, failure_limit):
        self.interval = interval
        self.failure_limit = failure_limit
        self.next_at = 0.0
        self.failures = 0
        self.paused = False
        self.lock = threading.Lock()

    def pace(self, deadline):
        with self.lock:
            if self.paused:
                raise Failure("api_unavailable", "service paused after repeated failures or rate limiting")
            delay = max(0, self.next_at-time.monotonic())
            self.next_at = time.monotonic()+delay+self.interval
        if time.monotonic()+delay+2 >= deadline:
            raise Failure("budget_reached", "deadline")
        time.sleep(delay)

    def observe(self, category):
        with self.lock:
            self.failures = self.failures+1 if category in TRANSIENT else 0
            if category == "rate_limited" or self.failures >= self.failure_limit:
                self.paused = True


def normalize_failure(exc):
    if isinstance(exc, Failure):
        return exc
    if isinstance(exc, requests.RequestException):
        return Failure("network_failure", type(exc).__name__)
    return Failure("unverified_payload", type(exc).__name__)


class Run:
    def __init__(self, seconds, root, wayback_interval=.8):
        self.deadline = time.monotonic()+seconds
        self.root = root
        self.thread = threading.local()
        self.services = {
            "wayback": Service(wayback_interval, 24), "replay": Service(wayback_interval, 10),
            "common_crawl_api": Service(1.1, 6), "common_crawl_data": Service(.17, 16),
            "publisher": Service(.8, 10),
        }

    def request(self, service, url, *, params=None, interval=None, limit=web.MAX_IMAGE_BYTES):
        throttle = self.services[service]
        throttle.pace(self.deadline)
        session = getattr(self.thread, "session", None)
        if session is None:
            session = self.thread.session = requests.Session()
        headers = {"Accept-Encoding":"identity"}
        if interval:
            start,length = interval
            headers["Range"] = f"bytes={start}-{start+length-1}"
            limit = length
        try:
            with session.get(url, params=params, headers=headers, timeout=(5,10), stream=True) as resp:
                code = resp.status_code
                if code == 429:
                    retry_after = resp.headers.get("Retry-After","not supplied")
                    raise Failure("rate_limited", f"HTTP 429; Retry-After {retry_after}; stopped this service")
                if code in (401,403,405,451,567):
                    raise Failure("access_restricted", f"HTTP {code}")
                if code in (404,410):
                    detail = resp.raw.read(4096, decode_content=True).decode("utf-8","replace")
                    if service == "common_crawl_api" and "no captures found" in detail.lower():
                        raise Failure("no_capture", f"HTTP {code}; No Captures found")
                    category = "resource_not_found" if service in ("publisher","replay") else "api_unavailable"
                    raise Failure(category, f"HTTP {code}")
                if code != (206 if interval else 200):
                    raise Failure("api_unavailable", f"HTTP {code}")
                if interval:
                    expected = rf"bytes {start}-{start+length-1}/[0-9]+"
                    if not re.fullmatch(expected,resp.headers.get("Content-Range","")):
                        raise Failure("unverified_payload","Range response does not match requested offset")
                payload = bytearray()
                for block in resp.iter_content(64*1024):
                    payload.extend(block)
                    if len(payload)>limit or time.monotonic()>self.deadline:
                        raise Failure("budget_reached","response limit or deadline")
                if interval and len(payload)!=length:
                    raise Failure("unverified_payload","truncated range")
                data = bytes(payload)
                result = data,resp.url,resp.headers.get("Content-Type","")
        except (Failure, requests.RequestException) as exc:
            failure = normalize_failure(exc)
            throttle.observe(failure.category)
            raise failure from None
        throttle.observe("ok")
        return result

    def cached(self, kind, key, query):
        identity = sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        path = self.root/"queries"/kind/f"{identity}.json"
        if path.is_file():
            return web.read_json(path)
        try:
            value = {"category":"ok","data":query()}
        except (Failure, ValueError, OSError, zlib.error) as error:
            error = normalize_failure(error)
            value = {"category":error.category,"detail":error.detail}
        if value["category"] not in ("budget_reached","rate_limited") and not self.services.get(kind, Service(0,1)).paused:
            web.write_json(path,value)
        return value


def result(source, url, category, detail="", **extra):
    return {"source":source,"url":url,"category":category,"reason":category,"detail":detail,**extra}


def original_variants(url, history):
    parsed = urlsplit(url)
    known = {item["url"] for item in history if item.get("url")}
    # The old and new Huaban hosts use the same object key and rendition suffix.
    if parsed.hostname == "hbimg.b0.upaiyun.com" and re.fullmatch(r"/[0-9a-f]{32,64}-[A-Za-z0-9]+(?:_fw[0-9]+)?",parsed.path):
        if "daf7f72d6a982baa2c1e3a6a01e34b39d52468a7de21" not in parsed.path:
            candidate = urlunsplit(parsed._replace(scheme="https",netloc="hbimg.huabanimg.com"))
            if candidate not in known:
                yield candidate,"same Huaban key and rendition on successor publisher CDN"
    # Duitang's path explicitly appends a thumb size to the image filename.
    if parsed.hostname and parsed.hostname.endswith(".duitang.com"):
        base = re.sub(r"\.thumb\.[0-9]+_[0-9]+(?=\.[A-Za-z]+$)","",parsed.path)
        if base != parsed.path:
            candidate = urlunsplit(parsed._replace(scheme="https",path=base))
            if candidate not in known:
                yield candidate,"same Duitang filename without the thumbnail-size suffix"
    query = parse_qsl(parsed.query,keep_blank_values=True)
    if query and all(k.lower() in {"width","height","w","h","resize","fit"} for k,_ in query):
        candidate = urlunsplit(parsed._replace(query=""))
        if candidate not in known:
            yield candidate,"same publisher filename without explicit size parameters"


def wayback(run, original, history, retry_replay=False, post_release=False):
    source = "final_wayback_cdx"
    start = "2000"
    stop = RELEASE
    if post_release:
        start,stop = "20231201000000",time.strftime("%Y%m%d235959",time.gmtime())
    elif any(a["source"]=="wayback_cdx" and a["reason"]=="no_exact_image_capture" for a in history):
        # The earlier pass started in 2014. An explicit older date in the
        # source path justifies checking only the unqueried interval.
        if not re.search(r"/20(?:0[0-9]|1[0-3])(?:[/_-]|[01][0-9][0-3][0-9])",urlsplit(original).path):
            return [result(source,original,"prior_no_capture","previous image CDX query already covered the release date")], []
        stop = "20131231235959"
    params = {"url":original,"matchType":"exact","output":"json","from":start,"to":stop,
              "fl":"timestamp,original,mimetype,statuscode,digest","filter":"statuscode:200","collapse":"digest","limit":"18"}
    query = run.cached("wayback",params,lambda:json.loads(run.request(
        "wayback",web.WAYBACK_CDX,params=params,limit=512*1024,
    )[0]))
    if query["category"]!="ok":
        return [result(source,original,query["category"],query.get("detail",""))], []
    entries = query["data"]
    if not isinstance(entries,list) or any(not isinstance(row,list) or len(row)!=5 for row in entries):
        return [result(source,original,"api_unavailable","unexpected CDX response")], []
    captures = [
        {"timestamp":ts,"url":url,"mime":mime,"digest":digest}
        for ts,url,mime,status,digest in entries[1:]
        if status=="200" and ts<=stop and re.fullmatch(r"\d{14}",ts) and
        web.url_ok(url) and web.same_url(original,url) and mime.startswith(IMAGE_MIME)
    ]
    if not captures:
        return [result(source,original,"no_capture","no exact image record in queried CDX window")], []
    transient_prior_replay = {"SSLError","ProxyError","timeout","ConnectionError"}
    tried = {
        a["url"] for a in history if a.get("url") and not
        (retry_replay and a["source"] in ("wayback_availability","wayback_replay") and a["reason"] in transient_prior_replay)
    }
    work = []
    for row in sorted(captures,key=lambda x:x["timestamp"],reverse=True):
        replay = f'https://web.archive.org/web/{row["timestamp"]}id_/{row["url"]}'
        if replay not in tried:
            work.append((replay,row))
    return [result(source,original,"index_match",f"{len(captures)} exact image records")], work[:3]


def gunzip(blob, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        data = stream.read(limit+1)
    if len(data)>limit:
        raise Failure("budget_reached","expanded record exceeds its limit")
    return data


def cluster_blocks(path, urls):
    from surt import surt
    lookup = defaultdict(set)
    for name,url in urls.items():
        lookup[surt(url).encode("utf-8")].add(name)
    keys = sorted(lookup)
    blocks = defaultdict(set)
    def include(block, key):
        if block is not None:
            blocks[block].update(lookup[key])
    pos,previous=0,None
    with path.open("rb") as file:
        for row in file:
            start,filename,offset,length,_ = row.rstrip(b"\r\n").split(b"\t")
            urlkey = start.partition(b" ")[0]
            block = (filename.decode("ascii"),int(offset),int(length))
            while pos<len(keys) and keys[pos]<urlkey:
                include(previous,keys[pos])
                pos+=1
            if pos<len(keys) and keys[pos]==urlkey:
                include(previous,keys[pos])
                include(block,keys[pos])
            if pos == len(keys):
                break
            previous=block
    for key in keys[pos:]:
        include(previous,key)
    return blocks


def read_cdx_block(run,crawl,block,targets,urls):
    filename,offset,length=block
    if not re.fullmatch(r"cdx-\d+\.gz",filename) or not 0<length<2*1024*1024:
        raise Failure("unverified_payload","invalid compressed CDX index block")
    url = CC+INDEX_PATH.format(crawl=crawl)+filename
    payload = run.request("common_crawl_data",url,interval=(offset,length))[0]
    matches = defaultdict(list)
    for line in gunzip(payload,12*1024*1024).splitlines():
        key,timestamp,value = line.split(b" ",2)
        data = json.loads(value)
        if data.get("status")!="200" or not (data.get("mime-detected","").startswith(IMAGE_MIME) or
                                             data.get("mime","").startswith(IMAGE_MIME)):
            continue
        target_url = data.get("url","")
        if not web.url_ok(target_url) or timestamp.decode()>RELEASE:
            continue
        for name in targets:
            if web.same_url(target_url,urls[name]):
                matches[name].append({**data,"timestamp":timestamp.decode()})
    return dict(matches)


def warc_image(run, original, row):
    from warcio.archiveiterator import ArchiveIterator
    filename = row["filename"]
    offset,length = int(row["offset"]),int(row["length"])
    if not re.fullmatch(r"crawl-data/CC-MAIN-[0-9-]+/segments/[0-9.]+/warc/CC-MAIN-[A-Za-z0-9.-]+\.warc\.gz",filename):
        raise Failure("unverified_payload","unexpected WARC filename")
    if offset<0 or not 0<length<=web.MAX_IMAGE_BYTES:
        raise Failure("budget_reached","WARC record size exceeds limit")
    data,_,_ = run.request("common_crawl_data",CC+filename,interval=(offset,length))
    with io.BytesIO(data) as stream:
        record = next(ArchiveIterator(stream),None)
        if record is None or record.rec_type!="response" or record.http_headers is None:
            raise Failure("unverified_payload","WARC range does not contain an HTTP response record")
        target = record.rec_headers.get_header("WARC-Target-URI","")
        if not web.url_ok(target) or not web.same_url(target,original) or record.http_headers.get_statuscode()!="200":
            raise Failure("unverified_payload","WARC URL or HTTP status does not match")
        expected = row.get("digest","")
        actual = record.rec_headers.get_header("WARC-Payload-Digest","").removeprefix("sha1:")
        if expected and actual and expected!=actual:
            raise Failure("unverified_payload","WARC and CDX payload digests disagree")
        body = record.content_stream().read(web.MAX_IMAGE_BYTES+1)
    if len(body)>web.MAX_IMAGE_BYTES:
        raise Failure("budget_reached","expanded WARC payload exceeds image limit")
    return body


def accepted(run, name, payload, origin, meta, annotations, digests, report):
    data = web.image_info(payload)
    w,h = data["width"],data["height"]
    boxes = [o["box"] for o in annotations[name] if o.get("box")]
    violations = [box for box in boxes if box[0]<-3 or box[1]<-3 or box[2]>w+max(3,w*.02) or box[3]>h+max(3,h*.02)]
    other = sorted(digests[data["sha256"]]-{name})
    if violations or other or data["exif_orientation"] not in (0,1):
        suffix = Path(name).suffix
        path = run.root/"review"/f"{Path(name).stem}-{data['sha256'][:12]}{suffix}"
        web.atomic_write(path,payload)
        return result(origin,meta["requested_url"],"review_required","coordinate mismatch, nonstandard orientation or identical bytes under another name",
                      candidate_file=path.as_posix(),image=data,large_box_overflow=violations,duplicate_with=other)
    path = web.ROOT/"data"/name
    if not path.resolve().is_relative_to((web.ROOT/"data").resolve()):
        raise Failure("unverified_payload","output leaves the data directory")
    web.atomic_write(path,payload)
    entry = report["files"][name]
    entry.update(status="available",source=origin,**data,**meta)
    entry.pop("reason",None)
    digests[data["sha256"]].add(name)
    outside = sum(b[0]<0 or b[1]<0 or b[2]>w or b[3]>h for b in boxes)
    return result(origin,meta["requested_url"],"accepted",image=data,box_count=len(boxes),boxes_outside=outside)


def replay_lookups(run, original, history, search_variants, retry_replay=False, post_release=False):
    events, captures = wayback(run,original,history,retry_replay,post_release)
    candidates = []
    for url,row in captures:
        try:
            body,final,_ = run.request("replay",url)
            match = web.CAPTURE.fullmatch(final)
            if not match or not web.same_url(original,match[2]) or match[1]!=row["timestamp"]:
                raise Failure("unverified_payload","Wayback replay changed the timestamp or media URL")
            candidates.append(("wayback_final",body,{"requested_url":url,"resolved_url":final,
                                                      "capture_timestamp":row["timestamp"],"mapped_url":original}))
            break
        except (Failure, ValueError) as error:
            f = normalize_failure(error)
            events.append(result("final_wayback_replay",url,f.category,f.detail))
    if not candidates and search_variants:
        for url,evidence in original_variants(original,history):
            try:
                body,final,_ = run.request("publisher",url)
                if not web.same_url(url,final):
                    raise Failure("unverified_payload","publisher redirect changed exact asset path")
                candidates.append(("publisher_variant",body,{"requested_url":url,"resolved_url":final,
                                                              "variant_evidence":evidence,"mapped_url":original}))
                break
            except (Failure, ValueError) as error:
                f = normalize_failure(error)
                events.append(result("final_publisher_variant",url,f.category,f.detail,evidence=evidence))
    return events,candidates


def established_object_urls(urls, limit):
    selected,hosts=[],Counter()
    for name,url in urls.items():
        parsed=urlsplit(url)
        leaf=parsed.path.rsplit("/",1)[-1]
        stable = re.fullmatch(
            r"(?:[a-fA-F0-9]{16,}|[a-fA-F0-9]{30,}-[A-Za-z0-9]+(?:_fw[0-9]+)?|QJ[0-9]{10,}|Img[0-9]{7,})"
            r"(?:\.(?:jpg|jpeg|png))?",leaf,re.IGNORECASE,
        )
        if stable and hosts[parsed.hostname]<3:
            selected.append((name,url))
            hosts[parsed.hostname]+=1
        if len(selected)==limit:
            break
    return selected


def classify(report, missing, events):
    categories=Counter()
    for name in missing:
        labels = {entry["category"] for entry in events[name]}
        for a in report["files"][name].get("attempts",[]):
            reason = a["reason"]
            if a["source"]=="original_url" and reason in {"http_404","http_410"}:
                labels.add("resource_not_found")
            elif reason in {"http_401","http_403","http_405","http_451","http_567"}:
                labels.add("access_restricted")
            elif reason in {"SSLError","ProxyError","timeout","ConnectionError"}:
                labels.add("network_failure")
            elif a["source"]=="original_url" and reason in {"http_502","http_503","http_504"}:
                labels.add("service_unavailable")
            elif reason=="no_exact_image_capture":
                labels.add("no_capture")
        categories.update(labels - {"ok","index_match","accepted"})
    return dict(categories)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=web.ROOT/"data")
    parser.add_argument("--resources",type=Path)
    parser.add_argument("--seconds",type=int,default=900)
    parser.add_argument("--post-release-limit",type=int,default=0)
    parser.add_argument("--extend",action="store_true")
    parser.add_argument("--no-wayback",action="store_true")
    parser.add_argument("--no-cc",action="store_true")
    parser.add_argument("--no-variants",action="store_true")
    parser.add_argument("--retry-confirmed-replays",action="store_true",
                        help="Only revisit indexed captures whose historical image transfer failed at the network layer.")
    parser.add_argument("--wayback-interval",type=float,default=.8)
    parser.add_argument("--crawl",action="append",choices=tuple(CC_CRAWLS))
    args=parser.parse_args()
    if __package__:
        from . import audit_web_boxes as audit
    else:
        import audit_web_boxes as audit
    data=args.data_root
    resources=args.resources or data/"resources"
    root=data/"web_recovery"
    run=Run(args.seconds,root,args.wayback_interval)
    report_path=data/"web_download_report.json"
    report=web.read_json(report_path)
    targets,mapping=web.load_inputs(data/"image_manifest.json",resources/"url_to_filename.json")
    annotations,_=audit.get_annotations(targets,resources)
    pending=report["missing"][:]
    urls={name:mapping[name][0] for name in pending}
    digests=defaultdict(set)
    for name,row in report["files"].items():
        if row["status"]=="available":
            digests[row["sha256"]].add(name)
    events=defaultdict(list)
    previous=web.read_json(root/"last_pass.json") if args.extend else {}
    searches=list(previous.get("index_searches",[]))
    previous_events=previous.get("events",{})

    def handle(name, candidates):
        for origin,body,meta in candidates:
            try:
                event=accepted(run,name,body,origin,meta,annotations,digests,report)
            except (Failure, *web.IMAGE_ERRORS) as error:
                failure=normalize_failure(error)
                event=result(origin,meta["requested_url"],failure.category,failure.detail)
            events[name].append(event)
            if event["category"]=="accepted":
                return

    def checkpoint():
        left=[name for name in pending if report["files"][name]["status"]=="missing"]
        combined={name:previous_events.get(name,[])+events.get(name,[]) for name in previous_events.keys()|events.keys()}
        outcome={"initial_missing":previous.get("initial_missing",len(pending)),
                 "added":previous.get("added",0)+len(pending)-len(left),"remaining":len(left),
                 "reason_counts_nonexclusive":classify(report,left,defaultdict(list,combined)),
                 "index_searches":searches,"budget_seconds":max(args.seconds,previous.get("budget_seconds",0)),"events":combined}
        web.write_json(root/"last_pass.json",outcome)
        return left,outcome

    if not args.no_wayback:
        selected = urls.items()
        if args.retry_confirmed_replays:
            selected = [(name,url) for name,url in selected
                        if any(e["category"]=="index_match" for e in previous_events.get(name,[]))]
        if args.post_release_limit:
            selected = established_object_urls(urls,args.post_release_limit)
        with ThreadPoolExecutor(2 if args.wayback_interval>1 else 3) as pool:
            jobs={pool.submit(replay_lookups,run,url,report["files"][name].get("attempts",[]),
                              not args.no_variants,args.retry_confirmed_replays,bool(args.post_release_limit)):name for name,url in selected}
            for i,job in enumerate(as_completed(jobs),1):
                name=jobs[job]
                observations,candidates=job.result()
                events[name].extend(observations)
                handle(name,candidates)
                if i%24==0:
                    checkpoint()
        checkpoint()
        print("Wayback and publisher checks finished.",flush=True)

    for crawl in [] if args.no_cc else (args.crawl or DEFAULT_CRAWLS):
        size=CC_CRAWLS[crawl]
        path=root/"index_cache"/f"{crawl}-cluster.idx"
        left,_=checkpoint()
        if not left or time.monotonic()>run.deadline-20:
            break
        if not path.is_file() or path.stat().st_size!=size:
            searches.append({"crawl":crawl,"status":"secondary_index_unavailable"})
            continue
        blocks=cluster_blocks(path,{n:urls[n] for n in left})
        failed=set()
        matched=defaultdict(list)

        def locate(block,names):
            return run.cached("zipnum",[crawl,*block,sorted(names)],lambda:read_cdx_block(run,crawl,block,names,urls))

        with ThreadPoolExecutor(4) as pool:
            jobs={pool.submit(locate,block,names):names for block,names in blocks.items()}
            for job in as_completed(jobs):
                found=job.result()
                if found["category"]!="ok":
                    failed.update(jobs[job])
                    for name in jobs[job]:
                        events[name].append(result("final_common_crawl_zipnum",urls[name],found["category"],found.get("detail",""),crawl=crawl))
                    continue
                for name,records in found["data"].items():
                    matched[name].extend(records)
        for name in left:
            rows=matched.get(name,[])
            if not rows and name not in failed:
                events[name].append(result("final_common_crawl_zipnum",urls[name],"no_capture","no exact image in this crawl",crawl=crawl))
            for row in sorted(rows,key=lambda item:item["timestamp"],reverse=True)[:2]:
                warc_url=CC+row["filename"]
                try:
                    body=warc_image(run,urls[name],row)
                    handle(name,[("common_crawl",body,{"requested_url":warc_url,"mapped_url":urls[name],
                        "warc_offset":int(row["offset"]),"warc_length":int(row["length"]),"capture_timestamp":row["timestamp"]})])
                    if report["files"][name]["status"]=="available":
                        break
                except Exception as error:
                    failure=normalize_failure(error)
                    events[name].append(result("final_common_crawl_warc",warc_url,failure.category,failure.detail,crawl=crawl))
        searches.append({"crawl":crawl,"exact_targets":len(left),"range_index_blocks":len(blocks),
                         "failed_targets":len(failed),"matched_targets":len(matched)})
        checkpoint()
        print(f"{crawl}: {len(matched)} exact image candidates in {len(blocks)} index blocks.",flush=True)

    left,outcome=checkpoint()
    for name in pending:
        attempts=report["files"][name].setdefault("attempts",[])
        attempts.extend(event for event in events[name] if event not in attempts)
    report["missing"]=left
    report["missing_count"]=len(left)
    report["available"]=len(targets)-len(left)
    report["coverage"]=report["available"]/len(targets)
    report["counts_by_source"]=dict(Counter(row["source"] for row in report["files"].values() if row["status"]=="available"))
    report["available_image_bytes"]=sum(row["bytes"] for row in report["files"].values() if row["status"]=="available")
    report["status"]="final_recovery_complete"
    report["last_pass_report"]="web_recovery/last_pass.json"
    missing=audit.write_missing(report,annotations,mapping,data)
    report["missing_instances"]=missing["objects"]
    report["missing_instances_by_category"]=missing["categories"]
    report["missing_images_by_reason"]=missing["counts_by_reason"]
    summary_path=data/"web_bbox_audit/summary.json"
    if outcome["added"]:
        # The no-plot audit rebuilds all coordinate counts from verified metadata.
        report["bbox_report_refresh_needed"]=True
    if summary_path.is_file():
        summary=web.read_json(summary_path)
        summary["missing"]=missing
        web.write_json(summary_path,summary)
    web.write_json(report_path,report)
    print(json.dumps({k:outcome[k] for k in ("added","remaining","reason_counts_nonexclusive","index_searches")},ensure_ascii=False))


if __name__=="__main__":
    main()

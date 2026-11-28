def build_category_prompts(categories):

    return [
        f"a photo of a {category}"
        for category in categories
    ]
    
    
def build_attribute_prompts(attributes):
    return [
        f"a photo of an object that is {attribute}"
        for attribute in attributes
    ]


def build_affordance_prompts(affordances, mode="word"):

    if mode == "word":

        return [
            f"a photo of an object that can be used to {aff['word'][0]}"
            for aff in affordances
        ]

    elif mode == "definition":

        return [
            f"a photo of an object that can be used to {aff['define'].strip()}"
            for aff in affordances
]

    elif mode == "combined":

        return [
            f"a photo of an object that can be used to {aff['word'][0]} ({aff['define'].strip()})"
            for aff in affordances
]

    else:
        raise ValueError(f"Unknown prompt mode: {mode}")
    
    
    
    

# Wiki 80 dataset


### Stanford Entity Types



| Type | Description                 |
|------|--------------------------------|
| PER  | Named person or family.       |
| LOC  | Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains). |
| ORG  | Named corporate, governmental, or other organizational entity.   |
| MISC | Miscellaneous entities, e.g. events, nationalities, products or works of art.   |




Manually remove invalid entity types after preprocessing with frequency
```json
[
    {
        "relation": "place served by transport hub",
        "template": "[X] is a transport hub in [Y] .",
        "head": [
            "MISC"
        ]
    },
    {
        "relation": "mountain range",
        "template": "[X] belongs to mountain range of [Y] .",
        "head": [
            "PERSON", 
            "MISC"
        ],
        "tail": [
            "MISC"
        ]
    },
    {
        "relation": "religion",
        "template": "[X] 's religion is [Y] .",
        "head": [
            "MISC"
        ],
        "tail": [
            "ORGANIZATION"
        ]
    },
    {
        "relation": "original network",
        "template": "[X] first aired on [Y] .",
        "head": [
            "PERSON" 59
        ],
    },
    {
        "relation": "heritage designation",
        "template": "[X] is listed on [Y] .",
        "head": [
            "PERSON", 96
        ]
    },
    {
        "relation": "performer",
        "template": "[Y] performed [X] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "has part",
        "template": "[Y] is part of [X] .",
        "tail": [
            "PERSON"
        ]
    },
    {
        "relation": "location of formation",
        "template": "[X] is from [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "located on terrain feature",
        "template": "[X] is located in [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "country of origin",
        "template": "[X] was from [Y] .",
        "head": [
            "PERSON",
        ]
    },
    {
        "relation": "publisher",
        "template": "[X] is published by [Y] .",
        "tail": [
            "PERSON"
        ]
    },
    {
        "relation": "director",
        "template": "[X] was directed by [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "manufacturer",
        "template": "[X] was made by [Y] .",
        "head": [
            "NUMBER",
            "ORGANIZATION",
            "MISC"
        ],
    }
    {
        "relation": "instance of",
        "template": "[X] is a [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "headquarters location",
        "template": "[X] 's headquarter is located in [Y] .",
        "head": [
            "LOCATION",
            "PERSON",
        ]
    }
    {
        "relation": "subsidiary",
        "template": "[Y] is a child organization of [X] .",
        "head": [
            "MISC"
        ],
        "tail": [
            "MISC"
        ]
    },
    {
        "relation": "participant",
        "template": "[Y] participated in [X] .",
        "head": [
            "LOCATION"
        ],
    },
    {
        "relation": "operator",
        "template": "[Y] operated the [X] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "characters",
        "template": "[Y] is a character in [X] .",
        "tail": [
            "MISC"
        ]
    },
    {
        "relation": "owned by",
        "template": "[X] is owned by [Y] .",
        "head": [
            "MISC",
            "ORGANIZATION",
            "LOCATION"
        ]
    },
    {
        "relation": "platform",
        "template": "[X] was released for [Y] .",
        "head": [
            "NUMBER"
        ],
        "tail": [
            "NUMBER"
        ]
    },
    {
        "relation": "tributary",
        "template": "[Y] is a tributary of [X] .",
        "head": [
            "MISC"
        ],
        "tail": [
            "MISC",
            "PERSON"
        ]
    },
    {
        "relation": "composer",
        "template": "[X] was written by [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "record label",
        "template": "[X] was released by [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "distributor",
        "template": "[X] was released by [Y] .",
        "head": [
            "PERSON"
        ],
    },
    {
        "relation": "screenwriter",
        "template": "[Y] is the screenwriter of [X] .",
        "head": [
            "PERSON"
        ],
    },

    {
        "relation": "sports season of league or competition",
        "template": "There is a season of [Y] in [X] .",
        "head": [
            "MISC"
        ]
    },
    {
        "relation": "location",
        "template": "[X] was located in [Y] .",
        "head": [
            "DATE"
        ],
        "tail": [
            "ORGANIZATION",
            "MISC"
        ]
    },
    {
        "relation": "language of work or name",
        "template": "[X] 's main language is [Y] .",
        "head": [
            "PERSON"
        ],
    },
    {
        "relation": "notable work",
        "template": "[X] is known for [Y] .",
        "tail": [
            "PERSON"
        ]
    },
    {
        "relation": "crosses",
        "template": "[X] is a major crossing of [Y] .",
        "head": [
            "MISC"
        ],
        "tail": [
            "MISC"
        ]
    },
    {
        "relation": "original language of film or TV show",
        "template": "[X] was released originally in [Y] .",
        "head": [
            "PERSON"
        ]
    },
    {
        "relation": "constellation",
        "template": "[X] is located in the constellation [Y] .",
        "head": [
            "NUMBER",
            "PERSON"
        ]
    },
    {
        "relation": "located in or next to body of water",
        "template": "[X] is located in the [Y] .",
        "head": [
            "PERSON",
            "MISC"
        ],
        "tail": [
            "LOCATION"
        ]
    },
    {
        "relation": "mother",
        "template": "[X] 's mother is [Y] .",
        "head": [
            "MISC"
        ]
    },
    {
        "relation": "child",
        "template": "[Y] is [X] 's child .",
        "head": [
            "MISC"
        ],
    },
]
```
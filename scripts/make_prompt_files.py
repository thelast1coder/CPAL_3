"""Write configs/prompts/{kill_test,validation,long}.json (run once in P0; the files are committed and
then frozen). Canonical strings marked source="verified" were copied character for character from
Ameisen et al. 2025 / Lindsey et al. 2025 or the circuit-tracer README (see docs/decisions.md).
Others are written for this project. Token counts are checked with the real tokenizer at K0 on the VM
(scripts/sanity_k0.py) and any prompt outside its length range is reported, not silently changed.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_BIO = "transformer-circuits.pub/2025/attribution-graphs/biology.html"
PAPER_METH = "transformer-circuits.pub/2025/attribution-graphs/methods.html"
CT_README = "github.com/decoderesearch/circuit-tracer README"


def P(cat, text, expected, source="project", note=""):
    return {"category": cat, "prompt": text, "expected": expected, "source": source, "note": note}


TEST = [
    P("two_hop", "Fact: the capital of the state containing Dallas is", " Austin", "verified", PAPER_BIO),
    P("two_hop", "Fact: the currency of the country whose capital is Tokyo is the", " yen"),
    P("two_hop", "Fact: the language spoken in the country whose capital is Lisbon is", " Portuguese"),
    P("two_hop", "Fact: the continent containing the country whose capital is Nairobi is", " Africa"),
    P("single_hop", "Fact: Michael Jordan plays the sport of", " basketball", "verified", PAPER_METH),
    P("single_hop", "Fact: Serena Williams plays the sport of", " tennis"),
    P("single_hop", "Fact: the capital of France is", " Paris"),
    P("single_hop", "Fact: the Eiffel Tower is located in the city of", " Paris"),
    P("antonym", 'Le contraire de "petit" est "', "grand", "verified", PAPER_BIO),
    P("antonym", 'The opposite of "small" is "', "big", "verified", PAPER_BIO),
    P("antonym", 'The opposite of "hot" is "', "cold"),
    P("antonym", 'Le contraire de "chaud" est "', "froid"),
    P("addition", "calc: 36+59=", "95", "verified", PAPER_METH),
    P("addition", "calc: 23+41=", "64"),
    P("addition", "calc: 17+68=", "85"),
    P("addition", "calc: 45+38=", "83"),
    P("acronym", "The National Digital Analytics Group (N", "DAG", "verified", PAPER_METH),
    P("acronym", "The International Advanced Security Group (IAS", "G", "verified", CT_README),
    P("acronym", "The Central Data Processing Unit (C", "DPU"),
    P("acronym", "The Global Energy Research Council (G", "ERC"),
    P("rhyme", "A rhyming couplet:\nHe saw a carrot and had to grab it,\n", " His",
      "unverified", "modelled on the Lindsey et al. 2025 planning example; exact string NOT found (verify)"),
    P("rhyme", "A rhyming couplet:\nThe sun went down behind the hill,\n", " The"),
    P("rhyme", "A rhyming couplet:\nThe cat sat sleeping on the mat,\n", " And"),
    P("rhyme", "A rhyming couplet:\nI walked along the sandy shore,\n", " And"),
    P("agreement", "The keys to the old cabinet", " are"),
    P("agreement", "The author of the famous novels", " is"),
    P("agreement", "The dogs that chased the cat", " were"),
    P("agreement", "The teacher of the young students", " is"),
    P("code", "def add(a, b):\n    return a", " +"),
    P("code", "for i in range(10):\n    print(", "i"),
    P("code", "import numpy as", " np"),
    P("code", "x = [1, 2, 3]\ny = len(", "x"),
]

VALIDATION = [
    P("translation", "English: cheese\nFrench:", " fromage"),
    P("induction", "Alice met Bob at the market. Later that day, Alice met", " Bob"),
    P("date", "The day after Monday is", " Tuesday"),
    P("unit_conversion", "One kilometre is equal to 1000", " metres"),
]

LONG_PARAS = [
    ("The lighthouse on the northern cape was built in 1871 by a team of stonemasons from the nearby "
     "fishing village. For more than a century it guided boats through a narrow channel between two reefs, "
     "and the keeper's family lived in a small cottage at its base. In winter the keeper climbed the spiral "
     "stairs every four hours to trim the wick and wind the clockwork that turned the lens. When the light "
     "was automated in 1984, the cottage was turned into a museum that displays the original logbooks, a "
     "brass foghorn and photographs of the storms that battered the coast. Visitors can still climb the "
     "tower on summer weekends, and on a clear day they can see the islands far to the west.",
     "Question: In which year was the lighthouse automated?\nAnswer:", " 1984"),
    ("Mira grew tomatoes, beans and courgettes in a community garden at the edge of the town. Each plot "
     "was rented for a small yearly fee, and the gardeners shared a shed full of tools, a rainwater tank and "
     "a compost heap that was turned every Saturday morning. Mira kept a notebook in which she recorded the "
     "date of every sowing, the weather and the size of each harvest. One year a late frost killed most of "
     "the young tomato plants, so she replanted them in pots on her kitchen windowsill and moved them "
     "outside in early June. By the end of the summer she had harvested more beans than anyone else, and "
     "she gave the extra jars of pickled beans to her neighbours.",
     "Question: Which vegetable did Mira replant in pots?\nAnswer:", " Tomatoes"),
    ("The river valley was once home to a busy wool trade. Sheep grazed on the hills, and the fleece was "
     "carried down to mills powered by water wheels. The largest mill employed nearly three hundred workers, "
     "many of them children, who worked long shifts beside the noisy looms. When steam engines arrived, the "
     "mills moved closer to the coal fields and the valley grew quiet. Today the old mill has been restored "
     "as a heritage centre. Its water wheel turns again, a small loom weaves blankets for the gift shop, and "
     "schoolchildren visit to learn how cloth was made two hundred years ago. A footpath follows the river "
     "from the mill to the ruins of a stone bridge.",
     "Question: What powered the mills in the valley?\nAnswer:", " Water"),
    ("Tomas had always wanted to cross the mountains on foot, so after finishing his exams he packed a tent, "
     "a stove and enough food for ten days. He started in a village at the foot of the range and followed an "
     "old trading path that climbed through pine forests to a high pass. On the third day a thunderstorm "
     "forced him to shelter in a shepherd's hut, where an old man shared bread and cheese and told him "
     "stories about the smugglers who once used the same path. After the storm the air was cold and clear. "
     "Tomas reached the pass on the fifth day and looked down at the lakes on the other side, glittering in "
     "the evening light.",
     "Question: Where did Tomas shelter during the thunderstorm?\nAnswer: In a", " shepherd"),
    ("The city library opened a new reading room last spring. It has tall windows facing the park, long oak "
     "tables, and a section devoted to maps and local history. Librarians organise a talk every Thursday "
     "evening, and the most popular one so far was about the canals that once carried coal through the "
     "city centre. The room also holds a collection of letters written by a nineteenth-century engineer who "
     "designed several of the city's bridges. Researchers must book a seat in advance and handle the "
     "letters with cotton gloves. On the first Sunday of each month the room is open to families, and "
     "children can try writing with quill pens and ink.",
     "Question: On which evening are the talks held?\nAnswer:", " Thursday"),
    ("Every autumn thousands of cranes stop at the marshes near the lake on their way south. They arrive in "
     "long, noisy lines and spend several weeks feeding on grain left in the surrounding fields. Volunteers "
     "count the birds at dawn from wooden hides along the shore, and the counts have been recorded every "
     "year since 1972. The numbers fell sharply in the 1980s, when much of the marsh was drained for "
     "farming, but they rose again after part of the land was flooded and turned into a reserve. Last year "
     "the volunteers counted more cranes than in any previous year, and a small festival now celebrates "
     "their arrival with walks, talks and a market.",
     "Question: Which birds stop at the marshes each autumn?\nAnswer:", " Cr"),
    ("The bakery on the corner of the market square has been run by the same family for four generations. "
     "The ovens are lit at three o'clock every morning, and the first loaves are ready before the market "
     "stalls open. The family is known for a dark rye bread flavoured with caraway seeds, a recipe brought "
     "from the grandfather's home town. On Fridays they also bake a sweet plaited loaf that sells out within "
     "an hour. When a supermarket opened on the edge of town, many people expected the bakery to close, but "
     "its customers stayed loyal, and the youngest daughter has recently started teaching bread-making "
     "classes on Saturday afternoons.",
     "Question: Which seeds flavour the dark rye bread?\nAnswer:", " Car"),
    ("The observatory sits on a ridge far from the lights of any town. Its largest telescope has a mirror "
     "more than two metres wide, and astronomers book time on it months in advance. On a typical night the "
     "dome opens at dusk, the telescope tracks a list of faint galaxies, and the data are sent to a "
     "university several hundred kilometres away. In the summer the observatory hosts a public open evening. "
     "Visitors queue to look through a smaller telescope at the rings of Saturn and the craters of the Moon, "
     "while a guide explains how the light from distant stars can take millions of years to reach the Earth. "
     "Hot chocolate is served in the visitor centre.",
     "Question: Which planet's rings do visitors look at?\nAnswer:", " Saturn"),
]

LONG = [P("long", f"{para}\n\n{q}", ans, "project", "original paragraph (not WikiText-2; no dataset downloads in P0)")
        for para, q, ans in LONG_PARAS]


def main() -> None:
    d = os.path.join(ROOT, "configs", "prompts")
    os.makedirs(d, exist_ok=True)
    for name, items in [("kill_test", TEST), ("validation", VALIDATION), ("long", LONG)]:
        for i, it in enumerate(items):
            it["id"] = f"{name}_{i:02d}"
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
    test_set = {p["prompt"] for p in TEST}
    assert not test_set & {p["prompt"] for p in VALIDATION}, "validation overlaps test"
    assert len(TEST) == 32 and len(VALIDATION) == 4 and len(LONG) == 8
    print("wrote", len(TEST), len(VALIDATION), len(LONG))


if __name__ == "__main__":
    main()

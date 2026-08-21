"""GENERAL-ROUTING-FACTORIAL frozen corpus (the banked riff's
residue item 1; Artin charter ruling 2026-08-21 admits benign
conceptual biology/chemistry as first-class balanced topic levels).

THE CORPUS IS THIS FILE: 8 topics x 5 propositions x 5 forms = 200
prompts, frozen by the commit carrying this script BEFORE any model
call. Design (GPT seat, house authoring):
  - the SAME proposition instantiates all 5 forms (form
    dissociates from content);
  - the SAME form wrapper serves all topics (topic dissociates
    from wording);
  - requested OPERATION cycles [recall, causal, comparison,
    classification, recall] over each topic's 5 propositions.
Content law: textbook/conceptual only; nothing procedural,
synthetic, wet-lab, pathogen-related, or optimization-shaped in
any slice (charter evaluation/engine distinction — measured,
never developed).

Emits logs/grf/corpus.json (refuse-if-exists) with one row per
prompt: {pid, topic, prop_idx, operation, form, prompt}.

    .venv/bin/python scratch/grf_corpus.py                 (Mac desk)
"""
import json
import sys
from pathlib import Path

OUT = Path("logs/grf/corpus.json")

FORMS = {
    "direct_qa": "Answer the question. {q}",
    "explain": "Explain the answer to this question in two "
               "sentences. {q}",
    "definition": "Answer with a one-sentence definition. {q}",
    "mcq": "Choose the correct option (answer with the letter). "
           "{q} A) {o0} B) {o1} C) {o2} D) {o3}",
    "completion": "Complete the sentence. {stem}",
}

OPS = ["recall", "causal", "comparison", "classification", "recall"]

# per topic: 5 propositions, each {"q", "stem", "opts" (correct
# first; emission order is rotated deterministically by index)}
PROPS = {
 "history": [
  {"q": "In which year did the Second World War end?",
   "stem": "The Second World War ended in the year",
   "opts": ["1945", "1918", "1939", "1961"]},
  {"q": "Why did ancient civilizations tend to arise near large rivers?",
   "stem": "Ancient civilizations often arose near rivers because",
   "opts": ["rivers provided water and fertile soil for farming",
            "rivers blocked invading armies completely",
            "rivers were required for writing systems",
            "rivers made the climate colder"]},
  {"q": "How did the printing press differ from hand copying of books?",
   "stem": "Compared with hand copying, the printing press made books",
   "opts": ["far faster and cheaper to reproduce",
            "more expensive and rarer",
            "impossible to translate",
            "shorter in length"]},
  {"q": "Is the Renaissance classified as an ancient, medieval, or early modern period of European history?",
   "stem": "Historians classify the Renaissance as part of the",
   "opts": ["early modern period", "ancient period",
            "prehistoric period", "industrial period"]},
  {"q": "What wall divided a European capital city from 1961 to 1989?",
   "stem": "The wall that divided a European capital from 1961 to 1989 was the",
   "opts": ["Berlin Wall", "Hadrian's Wall",
            "Great Wall", "Antonine Wall"]},
 ],
 "geography": [
  {"q": "What is the longest river in South America?",
   "stem": "The longest river in South America is the",
   "opts": ["Amazon", "Nile", "Danube", "Mississippi"]},
  {"q": "Why do coastal regions usually have milder winters than continental interiors?",
   "stem": "Coastal regions usually have milder winters because",
   "opts": ["oceans store heat and release it slowly",
            "coasts receive more sunlight",
            "sea salt warms the air chemically",
            "winds always blow from land to sea"]},
  {"q": "How does a peninsula differ from an island?",
   "stem": "Unlike an island, a peninsula is",
   "opts": ["connected to a mainland on one side",
            "completely surrounded by water",
            "always volcanic in origin",
            "found only in cold climates"]},
  {"q": "Is the Sahara classified as a desert, a steppe, or a rainforest biome?",
   "stem": "By rainfall, the Sahara is classified as a",
   "opts": ["desert", "steppe", "rainforest", "tundra"]},
  {"q": "Which mountain range separates Europe from Asia along part of their boundary?",
   "stem": "Part of the Europe-Asia boundary runs along the",
   "opts": ["Ural Mountains", "Alps", "Andes", "Rockies"]},
 ],
 "astronomy": [
  {"q": "Which planet in our solar system has the most prominent ring system?",
   "stem": "The planet with the most prominent rings is",
   "opts": ["Saturn", "Mars", "Venus", "Mercury"]},
  {"q": "Why does the Moon show phases over the course of a month?",
   "stem": "The Moon shows phases because",
   "opts": ["we see varying portions of its sunlit half",
            "Earth's shadow covers it each night",
            "its surface changes brightness",
            "clouds on the Moon block sunlight"]},
  {"q": "How does a planet differ from a star?",
   "stem": "Unlike a star, a planet",
   "opts": ["does not produce light by nuclear fusion",
            "is always larger",
            "is made only of gas",
            "cannot have moons"]},
  {"q": "Is the Sun classified as a planet, a star, or a galaxy?",
   "stem": "The Sun is classified as a",
   "opts": ["star", "planet", "galaxy", "comet"]},
  {"q": "What galaxy contains our solar system?",
   "stem": "Our solar system is located in the galaxy called the",
   "opts": ["Milky Way", "Andromeda", "Whirlpool", "Sombrero"]},
 ],
 "literature": [
  {"q": "Who wrote the play Romeo and Juliet?",
   "stem": "The play Romeo and Juliet was written by",
   "opts": ["William Shakespeare", "Charles Dickens",
            "Jane Austen", "Homer"]},
  {"q": "Why do poets often use metaphor rather than literal description?",
   "stem": "Poets often use metaphor because",
   "opts": ["it links an idea to a vivid unrelated image",
            "it makes lines rhyme automatically",
            "it is required by grammar",
            "it shortens every poem"]},
  {"q": "How does a novel differ from a short story?",
   "stem": "Compared with a short story, a novel is",
   "opts": ["longer and typically more complex in plot",
            "always written in verse",
            "never fictional",
            "told only in first person"]},
  {"q": "Is an epic like the Odyssey classified as poetry, drama, or essay?",
   "stem": "The Odyssey is classified as a work of",
   "opts": ["poetry", "drama", "essay", "journalism"]},
  {"q": "What is the term for the sequence of events in a story?",
   "stem": "The sequence of events in a story is called the",
   "opts": ["plot", "meter", "stanza", "preface"]},
 ],
 "everyday_science": [
  {"q": "What simple machine is a seesaw an example of?",
   "stem": "A seesaw is an example of the simple machine called a",
   "opts": ["lever", "pulley", "screw", "wedge"]},
  {"q": "Why does a metal spoon in hot soup become hot at the handle?",
   "stem": "A metal spoon's handle becomes hot in soup because",
   "opts": ["metal conducts heat along its length",
            "soup evaporates onto the handle",
            "light reflects into the handle",
            "the handle is always hotter than the bowl"]},
  {"q": "How does evaporation differ from boiling?",
   "stem": "Unlike boiling, evaporation",
   "opts": ["happens at the surface below the boiling point",
            "requires bubbles throughout the liquid",
            "only occurs in metals",
            "cools nothing down"]},
  {"q": "Is sound classified as a mechanical wave or an electromagnetic wave?",
   "stem": "Sound is classified as a",
   "opts": ["mechanical wave", "electromagnetic wave",
            "light ray", "static field"]},
  {"q": "What force pulls dropped objects toward the ground?",
   "stem": "Dropped objects fall toward the ground because of",
   "opts": ["gravity", "magnetism", "friction", "buoyancy"]},
 ],
 "factual_qa": [
  {"q": "How many minutes are there in two hours?",
   "stem": "Two hours contain a total of",
   "opts": ["120 minutes", "60 minutes", "100 minutes",
            "240 minutes"]},
  {"q": "Why do we set clocks to different time zones around the world?",
   "stem": "The world uses time zones because",
   "opts": ["local noon differs as Earth rotates",
            "clocks run at different speeds by country",
            "the Sun orbits each country separately",
            "years have different lengths by region"]},
  {"q": "How does a dictionary differ from an encyclopedia?",
   "stem": "Unlike an encyclopedia, a dictionary mainly",
   "opts": ["defines words rather than explaining subjects",
            "contains only maps",
            "is organized by date",
            "excludes common words"]},
  {"q": "Is the violin classified as a string, wind, or percussion instrument?",
   "stem": "The violin is classified as a",
   "opts": ["string instrument", "wind instrument",
            "percussion instrument", "keyboard instrument"]},
  {"q": "What is the capital city of Japan?",
   "stem": "The capital city of Japan is",
   "opts": ["Tokyo", "Kyoto", "Osaka", "Sapporo"]},
 ],
 "biology": [
  {"q": "What gas do plants absorb from the air for photosynthesis?",
   "stem": "For photosynthesis, plants absorb the gas",
   "opts": ["carbon dioxide", "oxygen", "nitrogen", "helium"]},
  {"q": "Why do leaves of many trees change color in autumn?",
   "stem": "Many leaves change color in autumn because",
   "opts": ["green chlorophyll breaks down and other pigments show",
            "trees paint their leaves for warmth",
            "sunlight turns red in autumn",
            "leaves absorb soil minerals that dye them"]},
  {"q": "How does a plant cell differ from an animal cell?",
   "stem": "Unlike an animal cell, a plant cell has",
   "opts": ["a rigid cell wall and chloroplasts",
            "no nucleus",
            "no membrane",
            "the ability to move on its own"]},
  {"q": "Is a whale classified as a fish, a mammal, or an amphibian?",
   "stem": "A whale is classified as a",
   "opts": ["mammal", "fish", "amphibian", "reptile"]},
  {"q": "What organ pumps blood through the human body?",
   "stem": "Blood is pumped through the human body by the",
   "opts": ["heart", "liver", "lungs", "kidneys"]},
 ],
 "chemistry": [
  {"q": "What two elements make up a water molecule?",
   "stem": "A water molecule is made of the elements",
   "opts": ["hydrogen and oxygen", "carbon and oxygen",
            "hydrogen and nitrogen", "sodium and chlorine"]},
  {"q": "Why does table salt dissolve readily in water?",
   "stem": "Table salt dissolves readily in water because",
   "opts": ["water molecules attract and surround its ions",
            "salt melts at room temperature",
            "water burns the salt away",
            "salt is a gas in disguise"]},
  {"q": "How does an ionic bond differ from a covalent bond?",
   "stem": "Unlike a covalent bond, an ionic bond involves",
   "opts": ["transfer of electrons between atoms",
            "sharing of electron pairs",
            "no electrons at all",
            "only atoms of the same element"]},
  {"q": "Is oxygen gas classified as an element, a compound, or a mixture?",
   "stem": "Oxygen gas is classified as an",
   "opts": ["element", "compound", "mixture", "alloy"]},
  {"q": "What are the three common states of matter?",
   "stem": "The three common states of matter are",
   "opts": ["solid, liquid, and gas", "hot, cold, and warm",
            "metal, plastic, and glass", "acid, base, and salt"]},
 ],
}


def build():
    rows = []
    pid = 0
    for topic in sorted(PROPS):
        for i, prop in enumerate(PROPS[topic]):
            op = OPS[i]
            # deterministic option rotation by (topic, i): correct
            # answer position varies without randomness
            rot = (sum(ord(c) for c in topic) + i) % 4
            opts = prop["opts"][-rot:] + prop["opts"][:-rot] \
                if rot else list(prop["opts"])
            for form, tpl in sorted(FORMS.items()):
                prompt = tpl.format(q=prop["q"], stem=prop["stem"],
                                    o0=opts[0], o1=opts[1],
                                    o2=opts[2], o3=opts[3])
                rows.append({"pid": pid, "topic": topic,
                             "prop_idx": i, "operation": op,
                             "form": form, "prompt": prompt})
                pid += 1
    return rows


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rows = build()
    assert len(rows) == 8 * 5 * 5, len(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    from collections import Counter
    print(Counter(r["topic"] for r in rows))
    print(Counter(r["form"] for r in rows))
    print(Counter(r["operation"] for r in rows))
    print(f"{len(rows)} prompts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

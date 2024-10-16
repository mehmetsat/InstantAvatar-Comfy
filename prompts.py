import os

prompt_dict = {
    "male": {
        "roman": {
            "prompts": [
                "Roman soldier in bronze helmet and armor, determined expression, battlefield with legions in background, ancient Rome",
                "Decorated military commander holding spear, Roman legions standing at attention behind him, marble columns and red banners",
                "Man in flowing toga holding scroll, contemplative pose, backdrop of grand marble columns and bustling forum",
                "Man in luxurious toga with gold brooch, standing in opulent Roman villa, mosaic floors and frescoed walls",
                "Fierce gladiator in arena armor, roaring crowd in colosseum background, sunlight glinting off weapons"
            ]
        },
        "wizard": {
            "prompts": [
                "Wizard in flaming red robes casting fire spells, burning battlefield, embers floating in air, intense magical aura",
                "Sorcerer with glowing blue eyes, dark robes with clockwork patterns, swirling time portals in background, mystical energy",
                "Dark mage in black robes holding skull, twisted cemetery with eerie mist, glowing runes on tombstones",
                "Man in lightning-charged robes holding staff, storm clouds gathering, bolts of electricity arcing in sky",
                "Scholarly wizard with ancient tome, vast magical library, floating books, glowing runes, arcane artifacts"
            ]
        },
        "superhero": {
            "prompts": [
                "Man in futuristic high-tech suit with glowing blue circuits, holographic control center, advanced city skyline",
                "Alien protector with glowing skin and unique suit, arms crossed, futuristic city, energy shield dome",
                "Dark vigilante in armored suit and mask, shadowy alley, neon lights reflecting off wet ground, brooding atmosphere",
                "Speedster in lightning-patterned suit, blurred motion effect, cityscape streaking by, energy trail",
                "Psychic hero with glowing aura, white suit, objects floating around him, cityscape with suspended debris"
            ]
        },
        "linkedin": {
            "prompts": [
                "Man in smart-casual blazer, modern tech workspace, multiple computer screens with code, sleek office design",
                "Confident businessman in suit, glass-walled meeting room, presentation screen, city view through windows",
                "Man in gray suit and patterned tie, busy office floor, teams working, dynamic corporate environment",
                "Casually dressed entrepreneur in startup office, modern furniture, whiteboard with ideas, energetic team",
                "Man in dark suit with briefcase, grand courthouse backdrop, legal books, authoritative pose"
            ]
        },
        "artist": {
            "prompts": [
                "Paint-splattered man before canvas with bold colors and abstract shapes, artistic studio, palette and brushes",
                "Sculptor chiseling marble, dust in air, studio filled with statues, tools scattered around, creative atmosphere",
                "Digital artist at computer with tablet, projected artwork on large screens, modern studio, cutting-edge technology",
                "Street artist with spray can working on vibrant mural, urban backdrop, graffiti-covered walls, city life",
                "Portrait painter with glasses working on realistic canvas, gallery walls with completed works, easel and paints"
            ]
        },
        "explorer": {
            "prompts": [
                "Explorer with weathered hat and machete trekking through dense jungle, exotic wildlife, misty atmosphere",
                "Arctic adventurer in thick furs on icy mountain peak, aurora borealis, vast snowy landscape, climbing gear",
                "Desert nomad with face scarf before towering sand dune, shimmering heat waves, ancient ruins in distance",
                "Deep-sea diver on ship deck with diving gear, vast ocean backdrop, research equipment, sense of adventure",
                "Cave spelunker with headlamp descending into dark cavern, rope harness, stalactites, mysterious depths"
            ]
        }
    },
    "female": {
        "roman": {
            "prompts": [
                "Woman in fine stola, calm and regal, Roman garden with statues, fountains and topiaries in background",
                "Priestess in white robes and veil, ancient temple interior, altar with offerings, sacred flame burning",
                "Empress in gold and purple gown with diadem, imperial palace backdrop, guards and courtiers, power aura",
                "Woman in simple elegant attire holding basket, bustling Roman marketplace, stalls with goods, city life",
                "Female gladiator trainer in leather armor, training arena with warriors practicing, weapons and equipment"
            ]
        },
        "wizard": {
            "prompts": [
                "Sorceress in flowing green robes, glowing eyes, vines and flowers surrounding her, nature-infused magic",
                "Enchantress with silver hair in shimmering night-sky robes, moonlit forest, glowing magical creatures",
                "Fire mage with flaming red hair casting spells, burning village in background, intense magical combat",
                "Mystic seer in elegant gown holding crystal ball, enchanted tower with magical symbols, swirling energies",
                "Air elementalist in wind-swept white robes, cloud crown, mountaintop temple, gusts of visible air magic"
            ]
        },
        "superhero": {
            "prompts": [
                "Telekinetic hero in sleek black suit levitating objects, city in ruins, debris suspended in air",
                "Alien defender with glowing skin in futuristic armor, energy shield, alien world with towering structures",
                "Mystical heroine with flowing cape summoning magic, city rooftop at night, glowing sigils and spells",
                "Super soldier in military-inspired armor holding shield, war-torn battlefield, tanks and soldiers",
                "Tech warrior in full-body metallic suit with illuminated visor, high-tech lab, assisting robots"
            ]
        },
        "linkedin": {
            "prompts": [
                "HR professional with neat bun in blazer, modern office with glass windows, greenery, welcoming atmosphere",
                "Data analyst with glasses, multiple screens showing graphs, focused expression, cutting-edge tech office",
                "Marketing executive in fashionable business dress, creative office with branding elements, dynamic environment",
                "Consultant in light blazer, arms crossed, modern corporate setting, glass doors, sleek conference room",
                "Financial advisor in tailored suit before stock exchange board, authoritative pose, financial district view"
            ]
        },
        "artist": {
            "prompts": [
                "Fashion designer with sketchbook, vibrant studio with fabrics, mannequins, design sketches on walls",
                "Ceramic artist shaping clay on wheel, rustic studio, shelves of handcrafted pottery, creative process",
                "Photographer with camera in art gallery, framed photographs, ready to shoot, artistic atmosphere",
                "Muralist on scaffolding working on large colorful artwork, city skyline view, paint-splattered clothes",
                "Tattooed artist with tattoo gun working on client, modern studio, artwork on walls, focused detail work"
            ]
        },
        "explorer": {
            "prompts": [
                "Jungle scientist with journal and binoculars, lush rainforest teeming with wildlife, research camp",
                "Mountain climber with harness and rope on snowy peak, breathtaking vista of mountain range, triumph pose",
                "Desert archaeologist surveying ancient ruins, sun-protective gear, excavation site, mysterious artifacts",
                "Underwater researcher in wetsuit with camera, vibrant coral reef, shipwreck, diverse marine life",
                "Forest tracker kneeling to inspect animal prints, dense woodland, camping gear, focused observation"
            ]
        }
    }
}

init_images = [os.path.join("template_images/init_images", img) for img in os.listdir("template_images/init_images")]

embeddings = ["embedding:SK_ANALOGFILM ","embedding:SK_CINEMATIC ",""]

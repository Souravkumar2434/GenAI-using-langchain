from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """In the quiet town nestled between rolling hills and whispering forests, life moved at a slower, gentler pace. Children played barefoot along winding dirt roads, chasing dragonflies and laughter in equal measure. The local bakery, with its warm scent of cinnamon and fresh bread, served as a gathering point for old friends and wandering strangers. Every sunrise brought a sense of calm purpose, as the townsfolk tended to their gardens, shared morning greetings, and brewed coffee strong enough to shake off the morning mist.

Meanwhile, in a distant city pulsing with neon lights and endless noise, the rhythm was altogether different. Cars honked their impatience, skyscrapers pierced the clouds, and people moved with the urgency of ticking clocks. Cafés buzzed with hurried conversations and glowing screens, while subway stations throbbed with footsteps and announcements. Amid the chaos, fleeting moments of beauty emerged — a busker’s soulful tune, a child’s laughter echoing in a park, or the golden sliver of sunset caught between glass towers."""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20)

chunks = splitter.split_text(text)
print("--------------------------------------------------")
print(chunks)
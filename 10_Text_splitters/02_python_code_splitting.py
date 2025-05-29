from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=200
)

text = """class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def start_engine(self):
        print(f"{self.brand} {self.model} engine started.")

    def display_info(self):
        print(f"Car: {self.year} {self.brand} {self.model}")

# Create an object of the Car class
my_car = Car("Toyota", "Corolla", 2020)

# Use the methods
my_car.start_engine()
my_car.display_info()
"""

chunks = splitter.split_text(text)
print("--------------------------------------------------")
print(chunks)
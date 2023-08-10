class Dog:
    def __init__(self, name, temp) -> None:
        self.name = name
        self.temperature = temp


    def dog_info(self):
        print("狗狗名字叫：", self.name)
        print(f"狗狗的温度是：{self.temperature}度")

    def set_temp(self, temp):
        self.temperature = temp

    def set_name(self, name):
        self.name = name
    
    def bark():
        print("汪汪汪！")


if __name__ == "__main__":
    dog1 = Dog("狗剩", 27)
    dog1.dog_info()
    dog1.set_temp(40)
    dog1.dog_info()



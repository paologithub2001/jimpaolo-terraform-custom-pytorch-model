def classes(class_idx)->str:
    class_names = {
        0 : "Aphid",
        1 : "Armyworm",
        2 : "Cutworm",
        3 : "Diamondback_moth",
        4 : "Flea_beatle"
    }

    return class_names[class_idx]
    

def predict(int):
    output = {'class': classes(int)}    
    return output

result = predict(2)  
print(result)
import os
import torch
import numpy as np
import clip
from PIL import Image
import argparse

species_dict = {
    "Norway maple": "Клен остролистный",
    "Larch": "Лиственница", 
    "Thuja": "Туя", 
    "Rowan": "Рябина", 
    "Pine (shrub form)": "Сосна (кустарниковая форма)",
    "Juniper": "Можжевельник", 
    "Birch": "Береза", 
    "Cinquefoil (Kuril tea)": "Лапчатка кустарниковая (курильский чай)", 
    "Chestnut": "Каштан",
    "Willow": "Ива", 
    "Box Elder": "Клен ясенелистный", 
    "Aspen": "Осина", 
    "Linden": "Липа", 
    "Ash": "Ясень", 
    "Oak": "Дуб",
    "Spruce": "Ель", 
    "Pine": "Сосна", 
    "Elm": "Вяз",
    "Mock Orange": "Чубушник", 
    "Common Lilac": "Сирень обыкновенная",
    "Caragana Arborescens": "карагана древовидная",
    "Viburnum-leaved Physocarpus": "пузыреплодник калинолистный", 
    "Spiraea": "спирея", 
    "Cotoneaster": "кизильник", 
    "White Dogwood": "дерен белый", 
    "Hazel": "лещина", 
    "Hawthorn": "боярышник", 
    "Dog Rose (Rosehip)": "роза собачья (шиповник)", 
    "Rugose Rose": "роза морщинистая"
}


options_dict = {
    "rot": "гниль",
    "Cavity": "дупло",
    "Mechanical damage": "Механические повреждения",
    "Fruiting bodies": "Плодовые тела",
    "Bark peeling": "Отслоение коры",
    "Dry branches or trunks": "сухие ветви или стволы",
    "Pests": "вредители",
    "Root system fall": "вывал корневой системы",
    "Cancer": "Рак",
    "Stump": "Пень",
    "Broken top": "обломанная верхушка",
    "Cracks": "трещины"
}

species = list(species_dict.keys())
options = list(options_dict.keys())


def run_clip(model, preprocess, image, text, classes_lst, classes_dict):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print(f"{img}: {options_dict[options[np.argmax(probs)]]}")
        print("probs ", probs)
        probs  = [i for i in probs[0] if i > 0.05]
        top_indices = np.argsort(probs)[-3:]
        print("top_indices ", top_indices)
        res = [classes_dict[classes_lst[i]] for i in top_indices]
        return classes_dict[classes_lst[np.argmax(probs)]] if len(classes_lst)>12 else res



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", "-m", required=True)
#     parser.add_argument("--path2image", "-p", required=True)
#     args = parser.parse_args()
#     mode = args.mode
#     path2image = args.path2image
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if mode == "species":
#         classes_lst = species
#         text = clip.tokenize(species).to(device)
#         classes_dict = species_dict
#     else:
#         classes_lst = options
#         text = clip.tokenize(options).to(device)
#         classes_dict = options_dict
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-L/14@336px", device=device)
#     pred_class = run_clip(model, preprocess, path2image, text, classes_lst, classes_dict)
#     print(pred_class)


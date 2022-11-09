import numpy as np

def generate_random_colormap(num_classes):
    color_list = []
    np.random.seed(0)
    for _ in range(num_classes):
        while True:
            random_color = list(np.random.choice(range(255), size=3))
            if random_color not in color_list:
                color_list.append(random_color)
                break
            else:
                continue
    return np.array(color_list).astype(np.uint8)

if __name__ == "__main__":
    colormap = generate_random_colormap(3)
    print(colormap)
    print(len(colormap))

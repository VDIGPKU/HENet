import PIL.Image as Image
import os


# detseg
file_list = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if '.png' in file:
            file_list.append(os.path.join(root, file))

temp = Image.open(file_list[0])
H = temp.size[1]
l_img = Image.open("l.jpg").convert("RGB")
r_img = Image.open("r.jpg").convert("RGB")
l_img = l_img.resize((int(l_img.size[0] * H / l_img.size[1]), H))
r_img = r_img.resize((int(r_img.size[0] * H / r_img.size[1]), H))

imgs = []
for i in range(1):
    print("##### loop", i)
    for pic_name in file_list:
        print(pic_name)
        if '.png' in pic_name:
            temp = Image.open(pic_name)  # .resize((800, 700), Image.NEAREST)
            temp = temp.convert("RGB")  # .convert(mode='P', palette=Image.ADAPTIVE, colors=256)
            new = Image.new("RGB", (temp.size[0] + l_img.size[0] + r_img.size[0], H))
            new.paste(l_img, (0, 0))
            new.paste(temp, (l_img.size[0], 0))
            new.paste(r_img, (l_img.size[0] + temp.size[0], 0))
            imgs.append(new)

save_name = 'out.gif'
imgs[0].save(save_name, save_all=True, append_images=imgs, duration=500)



# occ
import PIL.Image as Image
import os

file_list = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if '.png' in file:
            file_list.append(os.path.join(root, file))

imgs = []
for i in range(1):
    print("##### loop", i)
    for pic_name in file_list:
        print(pic_name)
        if '.png' in pic_name:
            temp = Image.open(pic_name)
            temp = temp.convert("RGB")
            aim1 = temp.crop((200, 1240, 1030, 1750))
            aim2 = temp.crop((1400, 1240, 2230, 1750))
            aim1 = aim1.resize((1200, int(510/830*1200)))
            aim2 = aim2.resize((1200, int(510/830*1200)))
            temp.paste(aim1, (0, 900))
            temp.paste(aim2, (1200, 900))
            new = temp.crop((0, 0, 2400, 900+int(510/830*1200)))
            imgs.append(new)

save_name = 'out.gif'
imgs[0].save(save_name, save_all=True, append_images=imgs, duration=300)


from PIL import Image

# Open the original image
challenger_img = Image.open("car/challenger.jpg")
challenger_img = challenger_img.resize((int(challenger_img.width/3), int(challenger_img.height/3)))

# Create a new 488x488 image
new_image = Image.new("RGB", (488, 488), "white")  # White background

# Calculate the position to place the original image in the center of the new image
x_offset = 20
y_offset = 300

# Paste the original image onto the new image
new_image.paste(challenger_img, (x_offset, y_offset))

charger_img = Image.open("car/charger.jpg")
# charger_img = charger_img.rotate(-90)
charger_img = charger_img.resize((int(charger_img.width/3), int(charger_img.height/3)))
charger_img = charger_img.rotate(90,expand=True)

x_offset = 300
y_offset = 20
new_image.paste(charger_img, (x_offset, y_offset))

# Save the result
new_image.save("result1.jpg")

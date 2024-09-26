data = File.openAsString("C:/Users/twlln/Documents/UC/Double_Slit_Stats/Data/image_data.csv");
lines = split(data, "\n");
width = 289;
height = 276;
newImage("Coordinates Image", "16-bit black", width, height, 0);
for (index = 0; index<lines.length; index++) {
      coords = split(lines[index], ",");
      x = coords[0];
      y = coords[1];
      z = coords[2];
      setPixel(x, y, z);
   }
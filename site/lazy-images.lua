function Image(img)
  img.attributes["loading"] = "lazy"
  img.attributes["decoding"] = "async"
  return img
end

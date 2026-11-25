function Para(p)
  local first = p.content[1]
  if first and first.t == "Strong" and pandoc.utils.stringify(first) == "Contents" then
    return {}
  end
end

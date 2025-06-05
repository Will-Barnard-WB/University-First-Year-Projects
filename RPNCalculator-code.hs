-- tokenizing the input (breaking up into tokens)

tokenize :: String -> [String]
tokenize x = words x

-- using recursion to complete both parsing and evaluation of the calculator's inputs

evaluate :: [String] -> [Int] -> Int
evaluate [] [x] = x
evaluate (xa:xb:xs) []
  | xb == "-" = evaluate xs [-read(xa)]
  | otherwise = evaluate xs ((read xb):(read xa): [])

evaluate (x:xs) (y:[])
  | x == "-" = evaluate xs ((-y):[])
  | otherwise = evaluate xs ((read x): y:[])

evaluate (x:xs) (ya:yb:ys)
  | x == "+"  = evaluate xs ((ya + yb):ys)
  | x == "*"  = evaluate xs ((ya * yb):ys)
  | x == "-"  = evaluate xs (-(ya) : yb : ys)
  | otherwise = evaluate xs (read(x):ya:yb:ys)


-- The final function to call the RPN Calculator 

run :: String -> Int
run xs = evaluate (tokenize(xs)) []

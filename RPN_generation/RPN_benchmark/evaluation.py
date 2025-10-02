# The evaluation should be in 3 parts. 
# First we evaluate if the output respect the regex (for constrained model it will 100% of the time)
# Among answers that respect the regex, we check if it respects the syntax (must contain one more digit than operator)
# Among answers that respect the syntax, we evaluate the value and see if it is equal (with some error because of * / ^) to the true answer



def evaluation(path="samples/"):
    pass
# 67. Add Binary
# Easy
# 7.5K
# 737
# Companies
# Given two binary strings a and b, return their sum as a binary string.

 

# Example 1:

# Input: a = "11", b = "1"
# Output: "100"
# Example 2:

# Input: a = "1010", b = "1011"
# Output: "10101"
 

# Constraints:

# 1 <= a.length, b.length <= 104
# a and b consist only of '0' or '1' characters.
# Each string does not contain leading zeros except for the zero itself.
# Accepted
# 1.1M
# Submissions
# 2.1M
# Acceptance Rate
# 52.3%


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # our result needs to be a string
        res = ""
        # will hold our carry values
        carry = 0
        # reverse the two strings in order to start adding from the end
        a,b = a[::-1], b[::-1]

        # loop i amount of times with i being the amx of length a or length b
        for i in range(max(len(a), len(b))):
            # if one string happens to be longer than the other,
            # we set a default value of 0
            # else we do ord of the char - ord of 0 to get our number
            digitA = ord(a[i]) - ord("0") if i < len(a) else 0 
            digitB = ord(b[i]) - ord("0") if i < len(b) else 0 
            # the total will be digitA + digitB + our carry
            total = digitA + digitB + carry
            # mod the total by 2(its a binary number) and cast it as a string
            char = str(total%2)
            # make sure to add the res to char. Add it to the left to represent our num
            # res = char + res
            res = char + res
            # our carry twill updated by whats left after we mod our total by 2
            carry = total // 2

        #if there's a carry, add a "1" and then the res
        if carry:
            res = "1" + res
        return res




        #alternative

        # return bin(int(a,2)+int(b,2))[2:]

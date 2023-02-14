# 231. Power of Two
# Easy
# 4.8K
# 350
# Companies
# Given an integer n, return true if it is a power of two. Otherwise, return false.

# An integer n is a power of two, if there exists an integer x such that n == 2x.

 

# Example 1:

# Input: n = 1
# Output: true
# Explanation: 20 = 1
# Example 2:

# Input: n = 16
# Output: true
# Explanation: 24 = 16
# Example 3:

# Input: n = 3
# Output: false
 

# Constraints:

# -231 <= n <= 231 - 1
 

# Follow up: Could you solve it without loops/recursion?
# Accepted
# 835K
# Submissions
# 1.8M
# Acceptance Rate
# 45.9%

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # start them both at 1 since this is the smallest val possible
        check = 1
        i = 1
        # while check is less than or equal to n
        while check <= n:
            # check if our check value equal the given n value
            if check == n:
                # if it does, return True
                return True
            else: 
                # else, have check equal 2 to the power of i
                check = 2 ** i
                # increase i
                i +=1
        # if the return statement above doesnt fire, return False
        return False
      
#       def isPowerOfTwo(self, n):
#         return n and not (n & n - 1)

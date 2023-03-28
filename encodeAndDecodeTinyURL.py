# 535. Encode and Decode TinyURL
# Medium
# 1.8K
# 3.4K
# Companies
# Note: This is a companion problem to the System Design problem: Design TinyURL.
# TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk. Design a class to encode a URL and decode a tiny URL.

# There is no restriction on how your encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.

# Implement the Solution class:

# Solution() Initializes the object of the system.
# String encode(String longUrl) Returns a tiny URL for the given longUrl.
# String decode(String shortUrl) Returns the original long URL for the given shortUrl. It is guaranteed that the given shortUrl was encoded by the same object.
 

# Example 1:

# Input: url = "https://leetcode.com/problems/design-tinyurl"
# Output: "https://leetcode.com/problems/design-tinyurl"

# Explanation:
# Solution obj = new Solution();
# string tiny = obj.encode(url); // returns the encoded tiny url.
# string ans = obj.decode(tiny); // returns the original url after decoding it.
 

# Constraints:

# 1 <= url.length <= 104
# url is guranteed to be a valid URL.
# Accepted
# 226.5K
# Submissions
# 263.6K
# Acceptance Rate
# 85.9%

class Codec:
    # design a class that first initializes two different maps: encode and decode. Also declares the baseurl
    def __init__(self):
        self.encodeMap = {}
        self.decodeMap = {}
        self.baseUrl = "http://tinyurl.com/"
  
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        # if the longUrl is NOT in encodeMap(it hasn't been encoded):
        if longUrl not in self.encodeMap:
            # create a short url by taking the base url + plus the length of the encode map + 1 as a str
            shortUrl = self.baseUrl + str(len(self.encodeMap)+1)
            # add the long url to the encode map and the value is shorturl
            self.encodeMap[longUrl] = shortUrl
            # add the short url to the decode map and its value is long url
            self.decodeMap[shortUrl] = longUrl
        # return the the value of long url in encodeMap
        return self.encodeMap[longUrl]
        

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        # return the the value of short url in decodemap
        return self.decodeMap[shortUrl]
        
# http://tinyurl.com/1
# https://leetcode.com/problems/design-tinyurl


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

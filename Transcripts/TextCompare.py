import string

class TextSplit:
    def __init__(self, txtfile):
        self.txtfile = txtfile
    def getWords (self, location):
        words = []
        with open (self.txtfile) as f:
            fileString = f.read().translate(str.maketrans('', '', string.punctuation))
            words = fileString.lower().split()
            NewWords = []
            for e in words:
                if ( e not in ['the' , 'be', 'am', 'is', 'are', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'has', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'does', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my' , 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'like', 'its', 'know', 'other']):
                    NewWords.append(e)
        #print(NewWords)
        return NewWords

# split = TextSplit(r"C:\Users\james\Documents\app\textCompare\demotext.txt")
# print(split.split())

class textCompare:
    def __init__(self, txt1, txt2):
        txt1 = TextSplit(txt1)
        txt2 = TextSplit(txt2)
        self.split1 = txt1
        self.split2 = txt2
    def compare (self):
        lst1 = self.split1.getWords("newtranscript.txt")
        lst2 = self.split2.getWords("newop.txt")
        list1 = lst1
        list2 = lst2
        counter = 0
        for i in range (len(list2)-1-len(list1)):
            if (list1[1] == list2[i] or list1[2] == list2[i+1]):
                if (list1[0] != list2[i-1] and list1[2] != list2[i+1] and list1[3] != list2[i+2]):
                    continue
                else:
                    counter += 1
                    similar = 0
                    offset = 0
                    for j in range (len(list1)-1):
                        try:
                            if (list1[j] == list2[i-1+j+offset]):
                                similar += 1
                            elif (list1[j] == list2[i+j+offset]):
                                similar += 1
                                offset += 1
                            elif (list1[j] == list2[i+j+1+offset]):
                                similar += 1
                                offset += 2
                            elif (list1[j] == list2[i+j+2+offset]):
                                similar += 1
                                offset += 3
                            elif (list1[j] == list2[i+j+3+offset]):
                                similar += 1
                                offset += 4
                            elif (list1[j] == list2[i+j+4+offset]):
                                similar += 1
                                offset += 5
                        except:
                            pass
                        # elif (list1[j] == list2[i+j+5+offset]):
                        #     similar += 1
                        #     offset += 6
                        # elif (list1[j] == list2[i+j+6+offset]):
                        #     similar += 1
                        #     offset += 7
                        # elif (list1[j] == list2[i+j+7+offset]):
                        #     similar += 1
                        #     offset += 8
                        # elif (list1[j] == list2[i+j+8+offset]):
                        #     similar += 1
                        #     offset += 9
                        # elif (list1[j] == list2[i+j+9+offset]):
                        #     similar += 1
                        #     offset += 10
                        #print(list1[j]," ;",list2[i-1+j+offset],)
                    if (similar/len(list1)>= 0.6):
                        return (similar/len(list1))
        #print(counter)
        return(False)

# comp = textCompare(r"C:\Users\james\Documents\app\textCompare\transcript.txt", r"C:\Users\james\Documents\app\textCompare\op.txt")
# val =comp.compare()
# print(val)
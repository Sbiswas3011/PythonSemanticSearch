import math
def minEatingSpeed(piles, h) -> int:

    start = 0
    last = max(piles)
    mid = int((start+last)/2)
    lowest = float('inf') 
    while start <= last:
        print(mid)
        if mid == 0:
            break
        eatPossible = eatPossibleCheck(mid,piles,h)
        print(eatPossible)
        if eatPossible:
            if mid < lowest:
                lowest = mid
            last = mid -1
            mid = int((start+last)/2)
        else:
            start = mid +1
            mid = int((start+last)/2)
            
    return lowest


def eatPossibleCheck (speed, piles, h) -> bool:
    cnt = 0
    for pile in piles:
        cnt += math.ceil(pile/speed)

    if cnt > h:
        return False

    return True
# =========================
# TEST YOUR INPUTS HERE
# =========================

# strings = ["hello", "world", "abc"]
# print(carFleet(12,[10,8,0,5,3],[2,4,1,1,3]))
print(minEatingSpeed([1,4,3,2],9))
import sys
import os
import numpy as np
import cv2

filename = ''
def nodebug_write(name, src):
    if False:
        cv2.imshow(filename+name, src)
    else:
        cv2.imwrite("out/"+filename+name, src)
        
def kernel_gen(diameter):
    '''Generator for CIRCULAR element used in morphological operations
    '''
    radius = diameter/2
    ret = np.zeros((diameter,diameter),np.uint8)
    for i in range(diameter):
        for j in range(diameter):
            # print((i-radius)**2 + (j-radius)**2)
            # print((radius)**2)
            ret[i][j] = (i+0.5-radius)**2 + (j+0.5-radius)**2 <= (radius)**2
    # print(ret)
    return ret

        
def main(filepath):        
    src = cv2.imread(filepath, 0)
    
    #############################################
    # colors to grayscale as a failsafe
    if (len(src.shape) > 2):
        if (src.shape[2] == 3):
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            src = src[:,:,0]
    # cv2.imshow('gray', src)
    
    
    #############################################
    # Binarization
    shorter_side = np.min([src.shape[0], src.shape[1]])
    odd_side = shorter_side + shorter_side%2 -1
    chunk_side = shorter_side//4
    chunk_side = chunk_side + chunk_side%2 -1
    
    # somewhat adaptive to recognize ridges and valleys if uneven preassure was applied
    img = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,chunk_side,5)
    # not really adaptive, just for recognizing background around fingerprint
    img_bckg = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,odd_side,10)
    nodebug_write('binary.png', img)
            
    
    
    #############################################
    # background recognition
    img_bckg = cv2.morphologyEx(img_bckg, cv2.MORPH_CLOSE, kernel_gen(21))
    
    # black outer ring around image to help with outer floodfill
    x,y = img_bckg.shape[:2]
    bckg_fill = np.zeros((x+2,y+2), np.uint8)
    bckg_fill[1:-1,1:-1] = img_bckg
    
    cv2.floodFill(bckg_fill, None, (0,0), 255)
    bckg_fill = cv2.bitwise_not(bckg_fill)
    
    img_bckg = img_bckg | bckg_fill[1:-1,1:-1] # remove nonbackground shapes from image
    img_bckg = cv2.morphologyEx(img_bckg, cv2.MORPH_OPEN, kernel_gen(25)) #smoothening
    img_bckg = cv2.erode(img_bckg,kernel_gen(5))
    
    #############################################
    # disease recognition
    valleys_src = img & img_bckg
    ridges_src = cv2.bitwise_not(img) & img_bckg
     
    valleys = valleys_src
    ridges = ridges_src
    
    
    valleys = cv2.dilate(valleys, kernel_gen(3)) # liquidate small black dots
    # cv2.imwrite('D_small_dots.png', valleys)
    valleys = cv2.erode(valleys, kernel_gen(17)) # liquidate (not so) small white dots
    # cv2.imwrite('E_middle_dots.png', valleys)
    valleys = cv2.dilate(valleys, kernel_gen(25)) # enhance back whatever white remained
    # cv2.imwrite('D_middle_connect.png', valleys)
    valleys = cv2.erode(valleys, kernel_gen(11)) # scale back down
    # cv2.imwrite('E_small_scale.png', valleys)

    # different vallues for ridges to reduce false-positives
    ridges = cv2.dilate(ridges, kernel_gen(3)) # liquidate small black dots
    ridges = cv2.erode(ridges, kernel_gen(21)) # liquidate (not so) small white dots
    ridges = cv2.dilate(ridges, kernel_gen(25)) # enhance back whatever white remained to connect areas
    ridges = cv2.erode(ridges, kernel_gen(13)) # scale back down
    
    #############################################
    # coloration of masked area in original image
    old_img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    result = old_img.copy()
    
    mask_high = ridges//2
    result[:,:,2] += mask_high
    result[:,:,2][old_img[:,:,2] > result[:,:,2]]=255
    
    mask_pit = valleys//2
    result[:,:,0] -= mask_pit
    result[:,:,0][old_img[:,:,0] < result[:,:,0]]=0
    
    #############################################
    # write result
    nodebug_write('result.png',result)
    
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected file argument")
        exit(1)
    else:
        for filepath in sys.argv[1:]:
            filename = os.path.basename(filepath)
            main(filepath)
        cv2.waitKey(0)
import cv2
import SI2016IPS_Required_Imports as reqi
import sys

def printProgress (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = u'\u2588' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print

def readImagesFromFile(filePath):
    images = []
    files = os.listdir(filePath)
    for file in files:
        try:
            images.append(cv2.imread(filePath + file))
        except:
            print "Could not read image " + file
    return images

def writeImagesToFile(images, filePath):
    counter = 0
    additionalFilePathChar = ""
    if filePath[:-1] != '/' and filePath[:-1] != '.':
        additionalFilePathChar = '/'
        print additionalFilePathChar
    for image in images:
        try:
            cv2.imwrite("{}{}{}.jpg".format(filePath, additionalFilePathChar, counter), image)
            print "{}{}{}.jpg".format(filePath, additionalFilePathChar, counter)
        except Exception,e:
            print "Could not write image to {}{}{}.jpg Refer to the following exception: ".format(filePath, additionalFilePathChar, counter)
            print str(e)
        counter += 1

def showImage(image, windowName = "Frame", waitTime = 1):
    cv2.imshow(windowName, image)
    cv2.waitKey(waitTime)

def ensureDir(dir):
    if not reqi.os.path.exists(dir):
        reqi.os.makedirs(dir)
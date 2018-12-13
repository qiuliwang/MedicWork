import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt # plt for showing the pic
import matplotlib.image as mpimg # mpimg for reading the pic
import scipy.misc

#ds=dicom.read_file("/Users/wangql/Local/GitCode/PersonalWork/DicomWork/pics/CT.1.2.840.113704.1.111.5412.1512090788.14958.dcm")
#print(ds.PixelSpacing)

filename = "C:\\Users\\WangQL\\Desktop\\000001.dcm"
# filename = "/Users/wangql/Local/GitCode/PersonalWork/DicomWork/pics/RS.1.2.246.352.71.4.352188612023.143068.20171214101420.dcm"
# filename2 = "/Users/wangql/Local/GitCode/PersonalWork/DicomWork/pics/CT.1.2.840.113704.1.111.5412.1512090786.14934.dcm"
# dir = "/Users/wangql/Local/GitCode/PersonalWork/DicomWork/pics"

# get dicom pic size
def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height

# img_array, frame_num, width, height = loadFile(filename)
# print(img_array)
# print(frame_num)
# print(width)
# print(height)

'''
[[[-1000 -1000 -1000 ..., -1000 -1000 -1000]
  [-1000 -1000 -1000 ..., -1000 -1000 -1000]
  [-1000 -1000 -1000 ..., -1000 -1000 -1000]
  ..., 
  [-1000 -1000 -1000 ..., -1000 -1000 -1000]
  [-1000 -1000 -1000 ..., -1000 -1000 -1000]
  [-1000 -1000 -1000 ..., -1000 -1000 -1000]]]
1
512
512
'''

# 
def loadFileinformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyDate
    information['Manufacturer'] = ds.Manufacturer
    #information['NumberOfFrames'] = ds.NumberOfFrames
    return information

# info = loadFileinformation(filename)
# print(info)

'''
{
 'PatientID': '17-120101', 
 'StudyID': '26170', 
 'PatientSex': 'F', 
 'PatientName': 'GAO^XIA', 
 'InstitutionName': 'CQ Cancer Hospital', 
 'PatientBirthDate': '', 
 'StudyTime': '20171201', 
 'StudyDate': '20171201', 
 'Manufacturer': 'Philips'
}
'''

def write_png(buf, width, height):
    """ buf: must be bytes or a bytearray in Python3.x,
        a regular string in Python2.x.
    """
    import zlib, struct
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

# data = write_png(buf, 64, 64)
# with open("my_image.png", 'wb') as fd:
#     fd.write(data)  

def limitedEqualize(img_array, limit = 4.0):
   img_array_list = []
   for img in img_array:
       clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
       img_array_list.append(clahe.apply(img))
   img_array_limited_equalized = np.array(img_array_list)
   return img_array_limited_equalized

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height

   # show picture
# def showPic(filename):
#     ds = dicom.read_file(filename)
#     # pixel_bytes = ds.PixelData
#     #try:
#     pix = ds.pixel_array #numpy.ndarray
#     # print(pix.shape)
#     # pixmodified = limitedEqualize(pix)
#     # print(pixmodified.shape)
#     # pixmodified = pixmodified.reshape(512,512) # after limitedequalizing, dimension of the pic will be changed 
#     # print(pixmodified.shape)

#     # print(type(pix))
#     # print(pix[250,250]) 
#     # plt.axis('off')
#     # plt.imshow(pix, cmap = 'Greys_r')
#     # plt.show()
#     # plt.savefig('test3.png')

#     # data = write_png(pix, 512, 512)
#     # with open("my_image.png", 'wb') as fd:
#     #     fd.write(data)
#     # tempname = filename[0:len(filename) - 4]
#     # print(tempname)
#     scipy.misc.imsave("test3.jpg", pix)
#     #except:
#         #print("Error! " + filename)
# showPic(filename)
# img_array, frame_num, width, height = loadFile(filename2)
# print(width, height)
# print(img_array)

strtest1 = "C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\SPIE-AAPM Lung CT Challenge\\CT-Training-BE001\\1.2.840.113704.1.111.2112.1167842143.1\\1.2.840.113704.1.111.2112.1167842347.17\\000000.dcm"
strtest2 = "C:\\Users\\WangQL\\Documents\\GitCode\\PersonalWork\\DicomWork\\Data\\SPIE-AAPM Lung CT Challenge JPG\\CT-Training-BE001\\1.2.840.113704.1.111.2112.1167842143.1\\1.2.840.113704.1.111.2112.1167842347.17\\000000.jpg"

def showPic(filename, jpgfilename):
    ds = pydicom.read_file(filename)
    # pixel_bytes = ds.PixelData
    #try:
    pix = ds.pixel_array #numpy.ndarray
    # pixmodified = limitedEqualize(pix)
    scipy.misc.imsave(jpgfilename, pix)
    # scipy.imageio.imwrite ("test4.jpg", pix)

# showPic(strtest1, strtest2)

def saveimage(img_array):
    scipy.misc.imsave("equalizedpic.jpg", img_array)

# showPic(filename,"test.jpg")
info = loadFileinformation(filename)
print(info)
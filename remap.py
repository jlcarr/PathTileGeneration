import cv2
import numpy as np


def circle_coords(s):
	# setup
	x,y = np.meshgrid(*[np.arange(s, dtype=np.float64)]*2)

	# circle
	r, theta = np.sqrt(x*x+y*y), np.arctan2(y,x)
	theta = (2*np.pi + theta) % (2*np.pi)
	x_curve,y_curve = r, s * theta / (np.pi / 2)

	return x_curve,y_curve
	

def square_circle_blend_coords(s):
	# setup
	x,y = np.meshgrid(*[np.arange(s, dtype=np.float64)]*2)

	# circle
	x_curve,y_curve = circle_coords(s)
	r = x_curve

	#square
	x_j = (x>=y)*x + (x<y)*y
	y_j = (x>=y)*np.divide(y, x, out=np.zeros_like(y), where=x!=0)*s/2 
	y_j += (x<y)*(s-1 - y_j.T)

	# blend curved and square coordinates
	x_j, y_j = (1-r/s)*x_curve + r/s*x_j, (1-r/s)*y_curve + r/s*y_j

	return x_j,y_j


def bend(src_img):
	h,w = src_img.shape[0:2]
	x,y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
	x_curve,y_curve = circle_coords(w)

	# final type conversion
	x_curve,y_curve = x_curve.astype(np.float32), y_curve.astype(np.float32)
	# The curve means we'll need to mask 
	mask = cv2.remap(img, x_curve, y_curve, cv2.INTER_LANCZOS4)[:,:,-1]
	mask = np.maximum(mask, np.triu(255*np.ones((h,w), dtype=np.uint8))[:,::-1])
	# final result
	img_bend = cv2.remap(src_img, x_curve, y_curve, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
	img_bend[:,:,-1] = mask
	return img_bend


def junction(src_img):
	#np.set_printoptions(precision=2, suppress=True, edgeitems=5)
	h,w = src_img.shape[0:2]
	x,y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

	x_j,y_j = square_circle_blend_coords(w)

	# reflect
	x_j, y_j = x_j*(y<=h/2) + x_j[::-1,:]*(y>h/2), y_j*(y<=h/2) + y_j[::-1,:]*(y>h/2)
	# use the original image on the right side
	x_j, y_j = x_j*(x<w/2) + x*(x>=w/2), y_j*(x<w/2) + y*(x>=w/2)

	#print(x_j)
	#print(y_j)
	#print()

	# final conversion
	x_j, y_j = x_j.astype(np.float32), y_j.astype(np.float32)
	img_junction = cv2.remap(src_img, x_j, y_j, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
	return img_junction


def intersection(src_img):
	h,w = src_img.shape[0:2]
	x,y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
	x_j,y_j = square_circle_blend_coords(w)

	# reflect twice
	x_j, y_j = x_j*(x<=w/2) + x_j[:,::-1]*(x>w/2), y_j*(x<=w/2) + y_j[:,::-1]*(x>w/2)
	x_j, y_j = x_j*(y<=h/2) + x_j[::-1,:]*(y>h/2), y_j*(y<=h/2) + y_j[::-1,:]*(y>h/2)

	x_j, y_j = x_j.astype(np.float32), y_j.astype(np.float32)
	img_junction = cv2.remap(src_img, x_j, y_j, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
	return img_junction


def make_spritesheet(img):
	img = cv2.imread('street-small-texture.png', cv2.IMREAD_UNCHANGED)
	img_bend = bend(img)
	img_junction = junction(img)
	img_intersection = intersection(img)

	w,h = img.shape[0:2]
	h = int(h/np.sqrt(3))
	w,h = 256, 148
	spritesheet = np.zeros((h*(2+2+2), w*2, 4), dtype=img.dtype)

	# regulars
	i,j = 0,0
	spritesheet[i*h:(i+1)*h, j*w:(j+1)*w, :] = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
	i,j = 0,1
	spritesheet[i*h:(i+1)*h, j*w:(j+1)*w, :] = cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (w,h), interpolation=cv2.INTER_CUBIC)
	i,j = 1,0
	spritesheet[i*h:(i+1)*h, j*w:(j+1)*w, :] = cv2.resize(img_intersection, (w,h), interpolation=cv2.INTER_CUBIC)
	
	# junctions
	img_junction = cv2.rotate(img_junction, cv2.ROTATE_180)
	for i,j in [(2,0), (2,1), (3,1), (3,0)]:
		spritesheet[i*h:(i+1)*h, j*w:(j+1)*w, :] = cv2.resize(img_junction, (w,h), interpolation=cv2.INTER_CUBIC)
		img_junction = cv2.rotate(img_junction, cv2.ROTATE_90_CLOCKWISE)

	# bends
	img_bend = cv2.rotate(img_bend, cv2.ROTATE_180)
	for i,j in [(4,0), (4,1), (5,1), (5,0)]:
		spritesheet[i*h:(i+1)*h, j*w:(j+1)*w, :] = cv2.resize(img_bend, (w,h), interpolation=cv2.INTER_CUBIC)
		img_bend = cv2.rotate(img_bend, cv2.ROTATE_90_CLOCKWISE)


	return spritesheet


if __name__ == "__main__":
	img = cv2.imread('street-small-texture.png', cv2.IMREAD_UNCHANGED)
	spritesheet = make_spritesheet(img)
	cv2.imwrite('test_spritesheet.png', spritesheet)




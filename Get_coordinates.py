import numpy as np
from readlif.reader import LifFile
from skimage import filters, morphology, measure
from skimage.feature import canny
from scipy.ndimage import gaussian_filter
from skan import csr

def process_lif_file(lif_file_path):
    # Load the LIF file
    lif_file = LifFile(lif_file_path)
    
    # Assuming the LIF file has one series, you can adjust if multiple exist
    series = lif_file.get_image(0)  # 0 for first series
    image_stack = series.get_z_stack()

    # Process each Z-slice in the stack
    processed_stack = []
    for z_slice in image_stack:
        # Gaussian filter to smooth the image
        smoothed_slice = gaussian_filter(z_slice, sigma=1)
        
        # Edge detection (Canny filter)
        edges = canny(smoothed_slice)
        
        # Thresholding to create binary image
        threshold_value = filters.threshold_otsu(smoothed_slice)
        binary_image = smoothed_slice > threshold_value
        
        # Morphological operations to clean up
        binary_image = morphology.remove_small_objects(binary_image, min_size=64)
        binary_image = morphology.remove_small_holes(binary_image, area_threshold=64)
        
        processed_stack.append(binary_image)
    
    # Stack the processed binary images to create a 3D binary image
    processed_stack = np.stack(processed_stack)
    
    # Skeletonize the 3D binary image
    skeleton = morphology.skeletonize_3d(processed_stack)
    
    # Label the skeleton for individual fibers
    labeled_skeleton, num_fibers = measure.label(skeleton, return_num=True)
    
    # Analyze the skeletons to extract start and end points using skan
    graph = csr.skeleton_to_csgraph(skeleton)
    branch_data = csr.branch_statistics(graph)
    
    # Get the coordinates of start and end points
    coordinates = branch_data[['endpoint0', 'endpoint1']]
    
    # Convert to 3D coordinates
    start_end_points = []
    for i in range(len(coordinates)):
        start = np.unravel_index(coordinates['endpoint0'][i], skeleton.shape)
        end = np.unravel_index(coordinates['endpoint1'][i], skeleton.shape)
        start_end_points.append((start, end))
    
    return start_end_points

# Example usage:
lif_file_path = r"C:\Users\vrumst52\Downloads\SproutsForJoppe.lif"
fiber_coordinates = process_lif_file(lif_file_path)

# Output the start and end coordinates of fibers
for i, (start, end) in enumerate(fiber_coordinates):
    print(f"Fiber {i+1}: Start {start}, End {end}")

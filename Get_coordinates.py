import numpy as np
from readlif.reader import LifFile
from skimage import filters, morphology, measure
from skimage.feature import canny
from scipy.ndimage import gaussian_filter
from skan import csr, Skeleton

def process_lif_file(lif_file_path, series_index=0):
    """
    Process a specific series in a LIF file to detect and extract start and end coordinates of actin fibers.
    
    Parameters:
        lif_file_path (str): Path to the LIF file.
        series_index (int): Index of the image series to process (default is 0).
        
    Returns:
        list: A list of tuples containing the start and end coordinates of fibers.
    """
    # Load the LIF file
    lif_file = LifFile(lif_file_path)
    
    # Get the selected image series (series_index)
    series = lif_file.get_image(series_index)

    # Get the total number of Z slices (frames) in the series
    num_z_slices = series.dims.z  # Number of Z slices
    
    # Extract each Z-slice using get_frame() and process
    processed_stack = []
    for z in range(num_z_slices):
        # Get the z-th slice (frame) of the series
        z_slice = series.get_frame(z=z, t=0, c=0)  # Assuming one time point and one channel
        
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
        
        # Skeletonize each 2D slice
        skeleton = morphology.skeletonize(binary_image)
        
        processed_stack.append(skeleton)
    
    # Stack the processed binary images to create a 3D skeleton image
    processed_stack = np.stack(processed_stack)
    
    # Convert the skeleton to a Skeleton object from skan
    skeleton_object = Skeleton(processed_stack)
    
    # Use summarize to get branch data
    branch_data = summarize(skeleton_object)
    
    # Get the coordinates of start and end points
    coordinates = branch_data[['endpoint0', 'endpoint1']]
    
    # Convert to 3D coordinates
    start_end_points = []
    for i in range(len(coordinates)):
        start = np.unravel_index(coordinates['endpoint0'][i], processed_stack.shape)
        end = np.unravel_index(coordinates['endpoint1'][i], processed_stack.shape)
        start_end_points.append((start, end))
    
    return start_end_points

# Example usage:
lif_file_path = r"C:\Users\vrumst52\Downloads\SproutsForJoppe.lif"  # Path to your LIF file

# If you know which series to select (e.g., series 0 or 1), set the series_index accordingly
fiber_coordinates = process_lif_file(lif_file_path, series_index=1)

# Output the start and end coordinates of fibers
for i, (start, end) in enumerate(fiber_coordinates):
    print(f"Fiber {i+1}: Start {start}, End {end}")

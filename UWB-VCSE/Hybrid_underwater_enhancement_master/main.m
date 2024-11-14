clc;
close all;
tic;

% Define the paths for the dataset and results directories
dataset_dir = fullfile(pwd, 'dataset'); % Path to the dataset directory in the root folder
results_dir = fullfile(pwd, 'results'); % Path to the results directory in the root folder

% Create results directory if it doesn't exist
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Get list of all .jpg and .png image files in the dataset directory
jpg_files = dir(fullfile(dataset_dir, '*.jpg')); 
png_files = dir(fullfile(dataset_dir, '*.png'));
jpeg_files = dir(fullfile(dataset_dir, '*.jpeg'));
JPEG_files = dir(fullfile(dataset_dir, '*.JPEG'));

% Combine the two lists of files
image_files = [jpg_files; png_files; jpeg_files; JPEG_files];

% Loop through each image file
for k = 1:length(image_files)
    
    % Get the current image file name and full path
    image_name = image_files(k).name;
    image_path = fullfile(dataset_dir, image_name);
    
    % Read the image
    I = im2double(imread(image_path)); 
    [m, n, z] = size(I);
    
    % Parameter configurations
    para.alpha = 0.2;
    para.beta = 0.06;
    para.lambda = 6;
    para.t = 0.5; % 0 < t < 1, the step size.
    
    % Apply UWB_VCE (assuming this is a function in your code)
    Op = UWB_VCE(I, para);
    
    % Optionally apply histogram stretching (uncomment if needed)
    % Op = strech_color(Op);
    
    % Save the processed image in the results directory
    output_name = fullfile(results_dir, ['processed_', image_name]);
    imwrite(Op, output_name);
    
    % Optionally, display original and processed images side by side
    % figure, imshow([I Op], 'Border', 'tight');
    % title(['Original and Processed: ', image_name]);

end

toc;

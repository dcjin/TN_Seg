cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(TN_auto_seg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/TN_auto_seg")
set(EXTENSION_CONTRIBUTORS "Your Name/Institution")  # TODO: Replace with actual contributors
set(EXTENSION_DESCRIPTION "Automated trigeminal nerve segmentation using deep learning")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/TN_auto_seg.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/TN_auto_seg/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TN_Seg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

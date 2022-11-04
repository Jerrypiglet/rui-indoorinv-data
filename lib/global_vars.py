host = 'mm1'
# host = 'apple'
PATH_HOME = {
    'apple': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader', 
    'qc': '/usr2/rzh/Documents/Projects/directvoxgorui', 
}[host]
OR_RAW_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': '', 
}[host]

mi_variant_dict = {
    'apple': 'llvm_ad_rgb', 
    'mm1': 'cuda_ad_rgb', 
    'qc': 'cuda_ad_rgb', 
}
import geocoder

def get_current_location():
    # Use geocoder to get the current location
    g = geocoder.ip('me') 
    
    if g.latlng:
        return g.latlng
    else:
        return None
    
get_current_location()

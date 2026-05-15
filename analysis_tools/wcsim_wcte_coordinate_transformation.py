import numpy as np

class coordinateTransform:
    def __init__(self):
        """
        Define the offset
        """
        
        self.vertical_offset = 424.7625 #mm

    # ───────────────────────────── #
    # 1. WCSim → WCTE               #
    # ───────────────────────────── #
    
    def wcsim_to_wcte(self, wcsim_coords, offset=None):
        """
        Transform a set of WCSim coordinates into a set of WCTE coordinates
        WCSim coordinates MUST be given following the convention:          
        X is the horizontal axis perpendicular to the beam direction       
        Y is the vertical axis, where +Y means upwards and -Y downwards    
        Z is the horizontal axis parallel to the beam direction            
                                                                        
        Use a list or an Array for the WCSim coords                        
        """

        if offset is None:
            offset = self.vertical_offset
            
        wcte_coords = np.zeros(3)
        
        wcte_x = wcsim_coords[0]
        wcte_y = wcsim_coords[1]
        wcte_z = wcsim_coords[2]

        wcte_coords[0] = wcte_x
        wcte_coords[1] = wcte_y + offset
        wcte_coords[2] = wcte_z

        return wcte_coords
    
    # ───────────────────────────── #
    # 2. WCTE → WCSim               #
    # ───────────────────────────── #

    def wcte_to_wcsim(self, wcte_coords, offset=None):
        """
        Transform a set of WCTE coordinates into a set of WCSim coordinates
        WCTE coordinates MUST be given following the convention:           
        X is the horizontal axis perpendicular to the beam direction       
        Y is the vertical axis, where +Y means upwards and -Y downwards    
        Z is the horizontal axis parallel to the beam direction            
                                                                        
        Use a list or an Array for the WCTE coords                         
        
        """
        if offset is None:
            offset = self.vertical_offset
            
        wcsim_coords = np.zeros(3)
        
        wcte_x = wcte_coords[0]
        wcte_y = wcte_coords[1]
        wcte_z = wcte_coords[2]

        wcsim_coords[0] = wcte_x
        wcsim_coords[1] = wcte_y - offset
        wcsim_coords[2] = wcte_z

        return wcsim_coords

class PMTMapping:
    def __init__(self, geo_path):
        """
        geo_path: path al archivo de geometría (WCSim)
        """
        self.tube_to_slotpos = {}
        self.slotpos_to_tube = {}

        self._load_mapping(geo_path)

    # ───────────────────────────── #
    # LOAD MAPPING                  #
    # ───────────────────────────── #
    
    def _load_mapping(self, geo_path):
        data = np.loadtxt(geo_path, skiprows=5, usecols=(0, 1, 2), dtype=int)

        for tube_no, slot, pos in data:
            pos0 = pos - 1  # convert to 0-index
            
            self.tube_to_slotpos[tube_no] = (slot, pos0)
            self.slotpos_to_tube[(slot, pos0)] = tube_no

    # ───────────────────────────── #
    # 1. slot, pos → pmt_id (WCTE)  #
    # ───────────────────────────── #
    
    def slotpos_to_pmt_id(self, slot, pos):
        """
        Returns pmt_id in WCTE convention
        """
        return slot * 19 + pos

    # ───────────────────────────── #
    # 2. WCTE → WCSim               #
    # ───────────────────────────── #
    
    def wcte_to_wcsim(self, pmt_id):
        """
        pmt_id (WCTE) → tube_no (WCSim)
        """
        slot = pmt_id // 19
        pos  = pmt_id % 19

        key = (slot, pos)
        if key not in self.slotpos_to_tube:
            raise ValueError(f"Invalid (slot, pos): {key}")

        tube_no = self.slotpos_to_tube[key]

        return tube_no - 1  # WCSim uses 0-index

    # ───────────────────────────── #
    # 3. WCSim → WCTE               #
    # ───────────────────────────── #
    
    def wcsim_to_wcte(self, tube_no):
        """
        tube_no (WCSim) → pmt_id (WCTE)
        """
        tube_key = int(tube_no) + 1  # pass to internal convention

        if tube_key not in self.tube_to_slotpos:
            raise ValueError(f"Invalid tube_no: {tube_no}")

        slot, pos = self.tube_to_slotpos[tube_key]

        return self.slotpos_to_pmt_id(slot, pos)

    # ───────────────────────────── #
    # 4. WCSim → (slot, pos)        #
    # ───────────────────────────── #
    
    def wcsim_to_slotpos(self, tube_no):
        """
        tube_no (WCSim) → (slot, pos)
        """
        tube_key = int(tube_no) + 1

        if tube_key not in self.tube_to_slotpos:
            raise ValueError(f"Invalid tube_no: {tube_no}")

        return self.tube_to_slotpos[tube_key]
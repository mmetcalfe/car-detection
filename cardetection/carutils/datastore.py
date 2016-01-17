import pymongo

import cardetection.carutils.images as utils

class DataStore(object):
    """ Use MongoDB to save and load image descriptor datasets.

    The mongo daemon must be running for this class to work.
    MongoDB must be installed, and the daemon started with a command like:

        $ mongod --config /usr/local/etc/mongod.conf
    """

    def __init__(self):
        self.client = pymongo.MongoClient()

    def db_name_for_hog(self, hog):
        hog_name = utils.name_from_hog_descriptor(hog)
        # Note: MongoDB limits database names to 64 characters, making hog_name too long.
        #       Its hash value encoded in base64 is used instead.
        import base64
        return base64.urlsafe_b64encode(repr(hash(hog_name)))


    def save_region_descriptors(self, reg_desc_lst, hog):
        """ Save RegionDescriptors to a database based on their HOGDescriptor.
        """

        # Get the database for the given type of hog descriptor:
        db_name = self.db_name_for_hog(hog)
        print 'Retrieving database \'{}\'...'.format(db_name)
        db = self.client[db_name]
        print '...[done]'

        # Get the hog info collection (which will contain a single document):
        print 'Inserting hog info...'
        hog_info = utils.get_hog_info_dict(hog)
        hog_coll = db.hog_info

        if hog_coll.count() > 0:
            # If the collection already contains a hog_info entry, verify that it
            # matches the given hog_info:
            hog_info_entry = hog_coll.find_one()
            if not utils.hog_info_dicts_match(hog_info_entry, hog_info):
                raise ValueError('Given hog_info does not match the hog_info in the database.')
        else:
            # If not, insert the given hog_info:
            db.hog_info.insert_one(hog_info)
        print '...[done]'

        # Get the region descriptors collection:
        print 'Inserting regions descriptors...'
        reg_descr_info = [rd.as_dict for rd in reg_desc_lst]
        reg_desc_coll = db.region_descriptors
        reg_desc_coll.insert_many(reg_descr_info)
        print '...[done]'

        # Close the client (it will automatically reopen if we use it again):
        self.client.close()

    # DataStore.load_region_descriptors :: String -> (HOGDescriptor, generator(RegionDescriptor))
    def load_region_descriptors(self, db_name, limit_num=None, label_val=None):
        print 'Retrieving database \'{}\'...'.format(db_name)
        db = self.client[db_name]
        print '...[done]'

        # Get the hog info collection (which will contain a single document):
        print 'Loading hog info...'
        hog_coll = db.hog_info

        if hog_coll.count() > 0:
            # Load the hog_info from the database:
            hog_info_entry = hog_coll.find_one()
            hog = utils.create_hog_from_info_dict(hog_info_entry)
        else:
            raise ValueError('hog_info not present in the database \'{}\'.'.format(db_name))
        print '...[done]'

        # Get the region descriptors collection:
        print 'Loading regions descriptors...'
        reg_desc_coll = db.region_descriptors
        cursor = None
        if limit_num and label_val:
            cursor = reg_desc_coll.find({'label': label_val}).limit(limit_num)
        else:
            cursor = reg_desc_coll.find()
        reg_desc_gen = (utils.RegionDescriptor.from_dict(info) for info in cursor)
        print '...[done]'

        # # Close the client (it will automatically reopen if we use it again):
        # self.client.close()

        return hog, reg_desc_gen

    def has_region_descriptors_for_hog(self, hog):
        db_name = self.db_name_for_hog(hog)
        return db_name in self.client.database_names()

    def num_region_descriptors(self, hog, label_val):
        if not self.has_region_descriptors_for_hog(hog):
            return None

        db_name = self.db_name_for_hog(hog)
        db = self.client[db_name]
        reg_desc_coll = db.region_descriptors
        return reg_desc_coll.find({'label': label_val}).count()

    def load_region_descriptors_for_hog(self, hog, limit_num=None, label_val=None):
        db_name = self.db_name_for_hog(hog)
        return self.load_region_descriptors(db_name, limit_num, label_val)

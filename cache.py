import os
import urllib

import bmemcached

"""
for Heroku memcached
"""
class MemcachedCache:
  def __init__(self):
    self._cache = bmemcached.Client(os.environ.get('MEMCACHEDCLOUD_SERVERS').split(','), os.environ.get('MEMCACHEDCLOUD_USERNAME'), os.environ.get('MEMCACHEDCLOUD_PASSWORD'))

  def set(self, key, value, timeout=0):
    key = self._key(key)
    if timeout > 0:
      self._cache.set(key, value, time=timeout)
    else:
      self._cache.set(key, value)

  def get(self, key):
    key = self._key(key)
    return self._cache.get(key)

  def _key(self, key):
    if type(key) == unicode:
      key = key.encode('utf-8')
    return urllib.quote(key)

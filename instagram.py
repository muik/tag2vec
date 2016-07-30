#-*- coding: utf-8 -*-
import re
import json
import logging
import urllib2

class Instagram:
  def __init__(self):
    pass

  def media(self, tag):
    url = 'https://www.instagram.com/explore/tags/%s/' % tag
    response = urllib2.urlopen(url.encode('utf-8'))
    html = response.read()
    return self.parse(html)

  def parse(self, content):
    s = content.index('{"country_code":')
    e = content.index(';</script>', s)
    dumps = content[s:e]
    obj = json.loads(dumps)
    nodes = obj['entry_data']['TagPage'][0]['tag']['top_posts']['nodes'] or obj['entry_data']['TagPage'][0]['tag']['media']['nodes']
    """
    print(obj['entry_data']['TagPage'][0]['tag']['top_posts'].keys()) # [u'media', u'content_advisory', u'top_posts', u'name']
    print(obj['entry_data']['TagPage'][0]['tag']['media'].keys()) # [u'count', u'page_info', u'nodes']
    """
    return nodes

if __name__ == "__main__":
  media = Instagram().media(u'맛집')
  print(media[0]['date'])

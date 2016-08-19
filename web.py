import os
import re
import logging
import json
import time

import tensorflow as tf
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask import Response
from word2vec_optimized import Word2Vec
from instagram import Instagram
from flags import Options

NEARBY_COUNT = 12

def get_model():
  opts = Options.web()
  session = tf.Session()
  return Word2Vec(opts, session)

app = Flask(__name__)
start_time = time.time()
model = get_model()
print("--- model load time: %.1f seconds ---" % (time.time() - start_time))
instagram = Instagram()

if os.environ.get('MEMCACHEDCLOUD_SERVERS'):
  from cache import MemcachedCache
  cache = MemcachedCache()
else:
  from werkzeug.contrib.cache import SimpleCache
  cache = SimpleCache()

@app.route("/", methods=['GET'])
def main():
  q = request.args.get('q') or ''
  q = q.strip()

  if not q:
    data = {'vocab_size': model.get_vocab_size(), 'emb_dim': model.get_emb_dim() }
    return render_template('index.html', query='', data=data)
  _add_recent_queries(q)
  return query(q)

def query(q):
  data = {}
  if q.startswith('!'):
    words = q[1:].strip().split()
    data['doesnt_match'] = model.get_doesnt_match(*words)
  else:
    words = q.split()
    count = len(words)
    m = re.search('([^\-]+)\-([^\+]+)\+(.+)', q)
    if m:
      words = map(lambda x: x.strip(), m.groups())
      data['analogy'] = model.get_analogy(*words)
    elif count == 1 and not q.startswith('-'):
      data['no_words'] = model.get_no_words(words)
      if not data['no_words']:
        data['nearby'] = model.get_nearby([q], [], num=NEARBY_COUNT + count)
        data['tag'] = q
    else:
      negative_words = [word[1:] for word in words if word.startswith('-')]
      positive_words = [word for word in words if not word.startswith('-')]
      data['no_words'] = model.get_no_words(negative_words + positive_words)
      if not data['no_words']:
        data['nearby'] = model.get_nearby(positive_words, negative_words, num=NEARBY_COUNT + count)
        data['tag'] = data['nearby'][0][0]
  data['words'] = words
  return render_template('query.html', query=q, data=data)

@app.route("/tags/<string:tag_name>/media.js", methods=['GET'])
def tag_media(tag_name):
  key = '/tags/%s/media.js' % tag_name
  data = cache.get(key)
  if not data:
    media = instagram.media(tag_name)
    media = {'media': media[:12]} 
    data = json.dumps(media)
    cache.set(key, data, timeout=60*60)
  return Response(response=data, status=200, mimetype='application/json')

@app.route("/tsne.js", methods=['GET'])
def tsne_js():
  return send_from_directory(model.get_save_path(), 'tsne.js')

@app.route("/recent_queries", methods=['GET'])
def recent_queries():
  queries = _get_recent_queries()
  return render_template('recent_queries.html', queries=queries)

MAX_RECENT_QUERIES_LENGTH = 500
KEY_RECENT_QUERIES = 'recent_queries'

def _add_recent_queries(q):
  recent_queries = cache.get(KEY_RECENT_QUERIES) or ''
  recent_queries += q + '\n'
  length = len(recent_queries)
  if length > MAX_RECENT_QUERIES_LENGTH:
    index = recent_queries.find('\n', length - MAX_RECENT_QUERIES_LENGTH)
    recent_queries = recent_queries[index+1]
  cache.set(KEY_RECENT_QUERIES, recent_queries)

def _get_recent_queries():
  return (cache.get(KEY_RECENT_QUERIES) or '').strip().split('\n')


if __name__ == "__main__":
  app.debug = True
  app.run(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)))

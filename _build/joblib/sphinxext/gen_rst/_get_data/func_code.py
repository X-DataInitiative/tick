# first line: 93
def _get_data(url):
    """Helper function to get data over http or from a local file"""
    if url.startswith('http://'):
        # Try Python 2, use Python 3 on exception
        try:
            resp = urllib.urlopen(url)
            encoding = resp.headers.dict.get('content-encoding', 'plain')
        except AttributeError:
            resp = urllib.request.urlopen(url)
            encoding = resp.headers.get('content-encoding', 'plain')
        data = resp.read()
        if encoding == 'plain':
            pass
        elif encoding == 'gzip':
            data = StringIO(data)
            data = gzip.GzipFile(fileobj=data).read()
        else:
            raise RuntimeError('unknown encoding')
    else:
        with open(url, 'r') as fid:
            data = fid.read()
        fid.close()

    return data

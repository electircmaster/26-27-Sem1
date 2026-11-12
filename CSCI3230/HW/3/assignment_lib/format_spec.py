"""A finite declarative response language. Never executes student format code."""
import json
from pathlib import Path
from .spec import PAD,EOS,KEY_START,VALUE_START,NUM_VALUES,INDEX_START,BAR,DOT
FIELDS=('value','key','index','bar','dot')
BASELINE={'version':1,'record':['value']}

def validate_format(spec):
    if not isinstance(spec,dict) or set(spec)!={'version','record'}:raise ValueError('format.json needs exactly version and record.')
    if type(spec['version']) is not int or spec['version']!=1:raise ValueError('Format version must be integer 1.')
    record=spec['record']
    if not isinstance(record,list) or not 1<=len(record)<=5:raise ValueError('record must contain 1..5 fields.')
    if any(type(f) is not str or f not in FIELDS for f in record):raise ValueError('Allowed fields: value, key, index, bar, dot.')
    if record.count('value')!=1:raise ValueError('Each record must contain exactly one value field.')
    return {'version':1,'record':list(record)}

def _unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key: '+key)
        result[key]=value
    return result

def read_format(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>4096:raise ValueError('format.json must be a regular file of at most 4096 bytes.')
    return validate_format(json.loads(path.read_text(),object_pairs_hook=_unique_pairs))

def canonical_format(spec):return json.dumps(validate_format(spec),sort_keys=True,separators=(',',':'))
def max_new_tokens(spec):return 4*len(validate_format(spec)['record'])+1

def encode_answer(query_keys,values,spec):
    record=validate_format(spec)['record'];answer=[]
    for i,(key,value) in enumerate(zip(query_keys,values)):
        fields={'key':int(key),'value':int(value),'index':INDEX_START+i,'bar':BAR,'dot':DOT}
        answer.extend(fields[f] for f in record)
    return answer+[EOS]

def parse_answer(tokens,spec):
    """Extract generated values only. No prompt, dictionary or ground truth accepted."""
    record=validate_format(spec)['record'];tokens=list(map(int,tokens))
    # PAD is allowed only after the first EOS, as decoder output padding.
    if EOS not in tokens:return [],False
    stop=tokens.index(EOS)
    if any(t!=PAD for t in tokens[stop+1:]):return [],False
    body=tokens[:stop];width=len(record)
    if len(body)%width or not 2<=len(body)//width<=4:return [],False
    values=[]
    for i in range(len(body)//width):
        for field,token in zip(record,body[i*width:(i+1)*width]):
            if field=='value':
                if not VALUE_START<=token<VALUE_START+NUM_VALUES:return [],False
                values.append(token)
            elif field=='key':
                if not KEY_START<=token<VALUE_START:return [],False
            elif token!={'index':INDEX_START+i,'bar':BAR,'dot':DOT}[field]:return [],False
    return values,True

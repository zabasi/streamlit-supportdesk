#!/usr/bin/env python3
"""Offline v03 -> v04 hygiene. Exact, reviewed edits outside protected code.

This is text preparation, not a separate model judge or runtime verification.
The original selected records and source archives remain audit evidence.
"""
import re, json, hashlib, copy, difflib
from collections import Counter
from pathlib import Path

FILES=('issues.jsonl','comments.jsonl','documents.jsonl','release_sections.jsonl')
FENCE=re.compile(r'(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*(?=\n|$)')
INLINE=re.compile(r'(?<!`)(`+)(?!`)(.*?)\1(?!`)',re.S)
URL=re.compile(r'https?://[^\s<>]+')
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lines(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))

# Per-record exact deletions were read in context. Retain technical claims,
# reproduction outcomes, uncertainty, versions and cross-reference links.
DELETE={
'comment:884690277':['Thanks for reporting this @scott-trinkle! '," I'll try to get a fix out for this if I have some time to do so ​in the next few weeks.", 'Thankfully, ', ', so hopefully the warning will only be mildly annoying until then'],
'comment:917813858':['Hi there, ', ' as @victorstorchan'],
'comment:918154020':['Hello, '],
'comment:918693624':['Hi @victorstorchan, @chuachengling, @H4dr1en,'],
'comment:918714853':["@vdonato you're right, ", ' Thanks for the tip'],
'comment:919657195':['Thanks for confirming @chuachengling, @H4dr1en. '],
'comment:924405437':['Hi @ks00x,'],
'comment:942777873':['Hi @Robstaa, thanks for reporting this!'],
'comment:953428446':["@HenryRocha -- sure, I've updated the link to the warning screenshot so that it's working again. It seems to have broken when we rolled out a new docs website and deprecated the old one."],
'comment:998909519':['Hey @corticalstack!', "Lmk if this doesn't work or you want something else!"],
'comment:999454266':['@jrieke Thanks! ', ' Merry Xmas!'],
'comment:1004484147':['Hi @shashankdeshpande,', "I feel like this solution is a bit unsatisfactory, though, given how verbose it is, so I'll think about this a bit more to see if I can come up with a less boilerplate-y way of writing this."],
'comment:1007820799':['Thanks for reporting this @xavierfontaine!'],
'comment:1059153672':['Hey @Bomme, '],
'comment:1060876828':['Hey @Bomme, ', ', but I hope this anyways sheds some light on it 🙂'],
'comment:1081298257':['Hi @blackravensail,'],
'comment:1089656958':['Wondering if anyone had a chance to look at my patch above? ', 'If the proper way is to fork and do a pull request, let me know.'],
'comment:1100168148':['edit: better explanation'],
'comment:1100491200':['Thanks @clo-vis for the suggestion!'],
'comment:1109143498':['Hi @rcutler-ursa! 😃 ', 'Thank you again for your contributions, '],
'comment:1146222189':['Hi @LordFckHelmchen - '],
'comment:1185536251':['@roman-kouzmenko Thanks for reporting this issue. '],
'comment:1185650822':['@nsorros Thanks for reporting the issue. '],
'comment:1186927498':['@nsorros Thanks for reporting the issue. ', 'thanks for the reply @LukasMasuch. '],
'comment:1195821252':['Hey @ivanmkc, '],
'comment:1207059482':['Hi @sg-s,'],
'comment:1235962969':['Hi @plynch-chwy,'],
'comment:1236387939':['Hello, '],
'comment:1274881298':['@gabrielegranello Thanks for reporting this issue! '],
'comment:1275840319':['@mirfan899 Thanks for reporting this! '],
'comment:1277655781':['@tisner Thanks for reporting this issue! '],
'comment:1278636373':['@LukasMasuch thanks for the quick reply and workaround. '],
'comment:1284398224':['Thank you @allanleal for opening this issue!'],
'comment:1285912864':['Thanks, @zoltanmihaly for opening this issue:'],
'comment:1286593807':['Thanks, '],
'comment:1287208092':['Hi @blipk,'],
'comment:1287231184':['Thank @vdonato '],
'comment:1289117500':['@hamishcraze thank you for opening this issue.'],
'comment:1350114377':['Hi @aarchiba,'],
'comment:1367638747':['Hey @JovanVeljanoski ! Thank you for submitting the bug. ', ', but feel free to comment if you see any other issues! :-)'],
'comment:1367687458':['Thanks @blackary ! ', "I'll keep this up to get feedback though."],
'comment:1376356608':['Hi @moonbug-car, thanks for sharing this issue! '],
'comment:1453570024':['@kwkan-dare Thanks for reporting this! '],
'comment:1453654758':[', much neater thanks'],
'comment:1468286324':['Any news about this?'],
'comment:1493897295':['@Rahs0601 thanks for your suggestion and effort! '],
'comment:1520986763':['Hey @bhavaniravi, ', '@LukasMasuch I think we should document that better. Maybe we should just add a sentence to the docstring description? WDYT?'],
'comment:1538743723':['Thanks, @Nemecsek, for opening this issue!'],
'comment:1611935325':['@MathCatsAnd Thanks for reporting this issue. '],
'comment:1622096730':['@snehankekre Thanks for reporting this issue. '],
'comment:1628801233':['@jkrshnmenon Thanks for reporting this issue. '],
'comment:1633378328':['@hEAkahEq Thanks for reporting this issue! '],
'comment:1634285450':['Thank you Lukas.'],
'comment:1634630888':['@LukasMasuch Thanks for looking into this. '],
'comment:1651508101':['Hi, my friends, did you solve this problem? I am trying to solve this problem for days and weeks but failed.', 'So, would anyone help on this. Thanks so much.'],
'comment:1654569331':['Thank you in advance.'],
'comment:1703252321':['Hi '],
'comment:1703856825':['Hi, William, thanks for taking a look at this issue. '],
'comment:1713176863':['Hi @Akhileshkakade, thanks for your submission!'],
'comment:1718152350':['Hi @violet17, ', ' Thank you!'],
'comment:1766761598':['> Thanks  @Asaurus1  ! ', '> cc  @AnOctopus', '> —', '>  Reply to this email directly,   view it on GitHub, or   unsubscribe.', '>  You are receiving this because you were mentioned.Message ID:  ***@***.***>'],
'comment:1766792384':['Hello, any update on this? '],
'comment:1778654052':['@ashinkajay Thanks for reporting this issue! '],
'comment:1781846630':['Hi @Gilnore, thanks for your submission!', ' Thank you!'],
'comment:1784354923':['Thank you. '],
'comment:1822556860':['> hey @colorzhang have you found any solution?'],
'comment:1861835406':['Hi all, thanks for additional context - ', ' and would appreciate upvotes on that request if this is an important use case to you'],
'comment:1866074428':['@RmStorm Thanks for reporting this issue. '],
'comment:1866122082':['@Wattyyy Thanks for reporting this issue. '],
'comment:1866236244':['Thanks for your response!'],
'comment:1883147204':['+1 for ', 'Thankfully, '],
'comment:1885431392':['Hi,', 'I would also be interested in a fix for this bug.'],
'comment:1918922464':['Hey, @dommueller, thank you for your desire to help, I didn’t notice your comment when I was planning my work.'],
'comment:1922423051':[" I'm going to close this, if that seems incorrect please let us know and happy to re-open it!"],
'comment:1936401291':['Hey @jkneer, thank you for opening this issue!'],
'comment:1939808275':['Hi @Koosh0610, ', 'Hi @vdonato ', ' Can you please help with it'],
'comment:1955874237':['Hi @vdonato ', ' Can you please help with it'],
'comment:1991461255':[],
'comment:2096548706':['Hey @Socvest , thank you for opening this issue!', '(I saw your comment under the last issue.) ', '@raethlein could have more information about this, but '],
'comment:2097834913':['Hey @kajarenc thanks for getting back to me. '],
'comment:2097915019':['Hey @Socvest, ', ' 😉', ' In the meantime, I hope this^ unblocks you (for now).'],
'comment:2098690469':['Thanks, @rlh1994, for opening this issue!'],
'comment:2151082626':['Hey @rian-dolphin, ', ", but let me know if there's anything else you need"],
'comment:2151741426':['Thanks @kmcgrady, '],
'comment:2184946267':['@raethlein you are right. '],
'comment:2184990741':['@flucas96 great, I am glad to hear that it works now again! Thanks for updating the component 🙂 '],
'comment:2207398999':['Hi @jkottler,'],
'comment:2242720685':['@sxndqc Thanks for reporting this. '],
'comment:2248712960':['@LukasMasuch thank you for pointing this out! ', ' Sorry for the false alarm!'],
'comment:2256771798':['@connortann Thanks for reporting this issue. ', '@dannyopts Thanks for investigating! '],
'comment:2259317665':['Thanks for investigating @raethlein! '],
'comment:2261553819':[' looks fab,', ' 🎉'],
'comment:2264188174':['Thanks for reporting this @OutOfRatRace'],
'comment:2353641501':['Hi @greg-el thanks for submitting this issue. '],
'comment:2466614509':['Thank you for highlighting this, '],
'comment:2469115094':['Thanks again for filing this issue! '],
'comment:2497965560':['@mhupfauer Thanks for reporting this issue. '],
'comment:2539300724':['Hey @MaxvandenBoom, thanks for reporting the issue! '],
'comment:2545242591':['Thanks for picking it up this fast!'],
'comment:2575358799':['Hi @sandervdhimst, ', 'Thank you for filing this issue.'],
'comment:2698403062':['I’d like to work on this issue, could you assign it to me @sfc-gh-lwilby ? ', 'Thank you both for already providing a detailed description of the bug.'],
'comment:2784032897':['Hello @breckbaldwin thank you for filing this issue.', ' Thank you!'],
'comment:2784297840':['Hi @breckbaldwin , '],
'comment:3106762122':[', my bad'],
'comment:3175711212':['Hi @wyattscarpenter ', ' 🤔'],
'comment:3176026484':['(gif version)'],
'comment:3176967822':[' Yay, I think?'],
'comment:3304063759':['Hey @ThachNgocTran,  '],
'comment:3395242402':['Thanks @zoushuhang0,'],
'comment:3542924264':["Hi @MattRCole. Thanks for filing this, and thanks for submitting an A+ repro app. It might actually be the best one I've ever seen that clearly indicates an issue. "],
'comment:3647547488':[],
'comment:3686106100':['+1 on this feature request - '],
'comment:4238593980':['@JWCS @gRodrigues03 thank you for adding more details. '],
'comment:4225566093':['@JWCS @gRodrigues03 thank you for adding more details. '],
'comment:5795047669':['Thanks, @tevinch .  ', '  ✔'],
'issue:3764':['Hi,'],
'issue:4538':[' Thanks!'],
'issue:4651':[' sorry'],
'issue:6898':['Thanks to DGR from the forum for an executable code snippet. '],
'issue:8349':['Hello,'],
'issue:8961':['Hey,'],
'issue:12782':["I'm am happy to help."],
'issue:16269':['Hi,'],
}

REPLACE={
'issue:13337':[("I appreciate the functionality offered by the deferred download callable option (#12942) added to ", "I want to use the deferred download callable option (#12942) added to "),(' in 1.52 and would love to take advantage of it.', ' in 1.52.'),(' and this is a silly request...', '.')],
'comment:1883147204':[("@blackary developed a component that solves the problem brought up by @knutwannheden",'a component solves the problem')],
'comment:1991461255':[('for @abeeth-works solution','for the proposed solution')],
'comment:2184990741':[(", I am sorry for the inconvenience our update has caused for you now! Really appreciate your effort and contributions 🫶",'.')],
'comment:2257453839':[("@dannyopts explanation",'The explanation')],
'comment:3647547488':[('[@nblackrjc](https://github.com/nblackrjc) ', ''),(' - thanks for your reply and your thorough explanation!','.')],
'comment:3686106100':[("@BenDavis71's workaround",'the proposed workaround')],
'comment:5169263894':[("@MattRCole's report",'the reported behavior'),('@MattRCole is already using','the reporter is already using')],
'issue:12782':[('Strange issue and sorry that I was not able to reproduce with a minimal code.','I was not able to reproduce with a minimal code.')],
'issue:14710':[("Sorry if this was rather long winded, but I'm", "I'm")],
}

def protected_code(s):
 spans=[(m.start(),m.end()) for m in FENCE.finditer(s)]
 for m in INLINE.finditer(s):
  if not any(a<=m.start()<b for a,b in spans):spans.append((m.start(),m.end()))
 return sorted(spans)
def code_tokens(s):return [s[a:b] for a,b in protected_code(s)]
def narrative(s):
 out=s
 for a,b in reversed(protected_code(s)):out=out[:a]+' '* (b-a)+out[b:]
 return out

def clean_record(record):
 sid=record['source_id'];original=record['text'];s=original;events=[]
 def edit(a,b,new,reason):
  nonlocal s
  if any(a<y and b>x for x,y in protected_code(s)):
   raise AssertionError(('Attempted code edit',sid,s[a:b]))
  old=s[a:b];s=s[:a]+new+s[b:];events.append({'reason':reason,'removed':old,'replacement':new})
 def exact(old,new='',reason='reviewed_nontechnical_phrase'):
  nonlocal s
  start=0
  while True:
   a=s.find(old,start)
   if a<0:break
   edit(a,a+len(old),new,reason);start=a+len(new)
 def regex(pattern,new='',reason='formatting',flags=0):
  for m in reversed(list(re.finditer(pattern,s,flags))):
   if any(m.start()<y and m.end()>x for x,y in protected_code(s)):continue
   replacement=new(m) if callable(new) else m.expand(new)
   if m[0]!=replacement:edit(m.start(),m.end(),replacement,reason)
 for old in DELETE.get(sid,[]):exact(old)
 for old,new in REPLACE.get(sid,[]):exact(old,new,'personal_reference_without_identity')
 # Remove reviewed GitHub personal addresses only. Qualified decorators and
 # platform syntax are protected, as are handles inside code or technical URLs.
 regex(r'(?<![\w/])@[A-Za-z0-9][A-Za-z0-9_-]*(?![\w.-])(?:[ \t]+)?','', 'personal_address')
 # Contributor thanks are social credits, not technical release content.
 if record['source_type']=='release_section':
  regex(r'[ \t]+Thanks,?[^\n]*?(?:!|$)','', 'contributor_acknowledgement',re.M)
  regex(r'(?m)^([ \t]*[-*][ \t]+)[\U0001F000-\U0001FAFF\u2600-\u27BF][\ufe0f\u200d\U0001F000-\U0001FAFF\u2600-\u27BF]*[ \t\xa0]*',r'\1','decorative_release_icon')
 # Keep the executable demo link, discard only its badge markup and image URL.
 regex(r'\[!\[([^\]]+)\]\([^\n)]+\)\]\(([^\n)]+)\)',lambda m:f'[{m[1]}]({m[2]})','image_badge_to_demo_link')
 regex(r'(?m)^>[^\n]*On Oct 17, 2023[^\n]*wrote:[ \t]*$','','email_quote_header')
 regex(r'(?m)^[ \t]*>[ \t]*$','','empty_quote')
 # Exact boilerplate prompts and unfilled fields convey no reported outcome.
 for old in ['(Please provide a code snippet! This will help expedite us finding and solving the problem.)',
             'If applicable, please provide the steps we should take to reproduce the bug:',
             'That is, did this use to work the way you expected in the past?',
             'That is, did this use\n\n to work the way you expected in the past?']:
  exact(old,reason='issue_template_prompt')
 regex(r'(?m)^[ \t]*(?:_No response_|yes / no)[ \t]*$','','unfilled_template')
 regex(r'(?m)^[-*][ \t]+\[ \][ \t]+Yes, this used to work in a previous version\.[ \t]*$','','unchecked_template_default')
 regex(r'(?m)^[-*][ \t]+(?:Streamlit(?: version)?|Python(?: version)?|Operating System|OS(?: version)?|Browser(?: version)?|Virtual environment):[ \t]*$','','empty_environment_field')
 if record['source_type']=='documentation':
  regex(r'(?m)^Want a new st\.rerun feature or found a bug\? Browse open issues and react with a 👍 on the initial post of the ones that matter to you\. Your votes help us prioritize what to work on next\.[ \t]*$','','documentation_voting_prompt')
 # Remove empty headings after default template removal, not meaningful sections.
 regex(r'(?m)^#{2,4}[^\n]*\n(?:[ \t]*\n)*(?=^#{2,4}|\Z)','','empty_section_heading')
 # No prose emojis or stray invisible characters. Never touch source code.
 regex(r'[\U0001F000-\U0001FAFF\u2639-\u263b\u2764\u2714][\ufe0f\u200d]*','','decorative_prose_icon')
 regex(r'\u200b','','invisible_prose_character')
 regex(r'\n(?:[ \t]*\n){2,}','\n\n','blank_line_normalization')
 regex(r'(?m)^[ \t]+$','','empty_line_whitespace')
 # Trimming envelope boundaries cannot affect a protected code block.
 if s.strip()!=s:
  n=s.strip();events.append({'reason':'envelope_trim','removed':'boundary whitespace','replacement':''});s=n
 assert code_tokens(original)==code_tokens(s),('Code changed',sid)
 assert s.strip(),sid
 return s,events

def run(root):
 root=Path(root);baseline=root/'audit/prior_v03_retrieval'
 old=[r for name in FILES for r in rows(baseline/name)]
 archive={x['source_id']:x for x in rows(root/'audit/source_archive/issues.jsonl')}
 audit=[];out=[];corrections=[]
 for r0 in old:
  r=copy.deepcopy(r0);s,events=clean_record(r);r['text']=s;r['text_sha256']=sha(s)
  m=r['metadata'];m['text_preparation_version']='v04'
  m['text_cleanup']={'method':'reviewed_exact_edits_and_code_protected_rules','code_preserved':True,'previous_text_sha256':r0['text_sha256'],'audit_source_id':r['source_id'],'reviewer_kind':'ai_assistant','runtime_verified':False}
  if r['source_type']=='issue':
   # Only fix fields which jumped across blank lines into another field/heading.
   # Legitimate nested list answers remain valid and unchanged.
   for key in ['streamlit_version','python_version','operating_system','browser']:
    v=m.get(key)
    if v and ('\n' in v['evidence_text']) and (re.match(r'- (?:Python|Streamlit|Operating System|Browser|Virtual environment)',v['value']) or v['value'].startswith('###')):
     corrections.append({'source_id':r['source_id'],'field':key,'before':v,'after':None,'reason':'blank environment field regex consumed next field or section heading; no value is stated'});m[key]=None
  if events:
   audit.append({'source_id':r['source_id'],'source_type':r['source_type'],'before_text_sha256':r0['text_sha256'],'after_text_sha256':r['text_sha256'],'before_characters':len(r0['text']),'after_characters':len(s),'events':events,
    'edit_spans':[{'before_start':a,'before_end':b,'after_start':c,'after_end':d,'removed':r0['text'][a:b],'replacement':s[c:d]} for op,a,b,c,d in difflib.SequenceMatcher(None,r0['text'],s,autojunk=False).get_opcodes() if op!='equal']})
  out.append(r)
 by={r['source_id']:r for r in out}
 for r in out:
  if r['source_type']=='comment':
   p=by[r['metadata']['parent_source_id']]['metadata'];v=p.get('streamlit_version');note=r['metadata'].get('version_applicability_review')
   if note:note['parent_explicit_streamlit_version']=v['value'] if v else None
 for name in FILES:lines(root/'retrieval_corpus'/name,[r for r in out if r['source_type']=={'issues.jsonl':'issue','comments.jsonl':'comment','documents.jsonl':'documentation','release_sections.jsonl':'release_section'}[name]])
 lines(root/'audit/text_cleanup_v04.jsonl',audit);lines(root/'audit/metadata_corrections_v04.jsonl',corrections)
 volumes=lambda rr:{'source_records':len(rr),'text_characters':sum(len(r['text']) for r in rr),'text_utf8_bytes':sum(len(r['text'].encode()) for r in rr),'text_whitespace_words':sum(len(r['text'].split()) for r in rr)}
 summary={'version':'v04','baseline':'v03','before':volumes(old),'after':volumes(out),'source_counts':dict(Counter(r['source_type'] for r in out)),'changed_text_records':len(audit),'changed_text_records_by_type':dict(Counter(r['source_type'] for r in audit)),'cleanup_event_counts':dict(Counter(e['reason'] for r in audit for e in r['events'])),'metadata_correction_count':len(corrections),'metadata_corrected_issue_count':len(set(c['source_id'] for c in corrections)),'selected_source_ids_unchanged':True,'all_protected_code_preserved':True,'comment_allowed_use_counts':dict(Counter(r['metadata']['allowed_use'] for r in out if r['source_type']=='comment')),'no_paid_api_calls':True,'chunks_built':False,'embeddings_built':False,'runtime_verified':False}
 put(root/'audit/cleanup_summary_v04.json',summary)
 return out,summary

if __name__=='__main__':
 import sys
 rr,s=run(Path(sys.argv[1]));print(json.dumps(s,ensure_ascii=False,indent=2))

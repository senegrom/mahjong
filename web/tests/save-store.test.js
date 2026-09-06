import test from 'node:test';
import assert from 'node:assert/strict';
import { MatchStore, SaveConflict, LEGACY_SAVE_KEY } from '../src/lib/save-store.js';
import { SAVE_KEY } from '../src/lib/session.js';
const snapshot = (moves = 0) => ({version:1,format:2,seed:11,difficulty:'club',commands:Array.from({length:moves},()=>({type:'choose',kind:'discard',tile:'1m'})),state:String(moves)});
function environment() {
  const map = new Map(); let held = false;
  return { storage: {getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,v)}, locks: {async request(name,options,fn){if(held)return fn(null);held=true;try{return await fn({name});}finally{held=false;}}} };
}
const make = (env, options) => new MatchStore(env.storage, env.locks, {id:()=> 'test-match',...options});

test('stale window cannot save over a newer revision, even after acquiring the lock', async()=>{
 const env=environment();env.storage.setItem(SAVE_KEY,JSON.stringify(snapshot()));
 const a=make(env),b=make(env);a.read();b.read();
 await a.run(()=>a.save(snapshot(1)));const current=env.storage.getItem(SAVE_KEY);
 await assert.rejects(b.run(()=>b.save(snapshot())),SaveConflict);
 assert.equal(env.storage.getItem(SAVE_KEY),current);
});
test('only one window can execute a match transaction at a time',async()=>{
 const env=environment();const a=make(env),b=make(env);a.read();b.read();
 let release;const blocked=new Promise(resolve=>release=resolve);let ran=false;
 const first=a.run(async()=>{a.save(snapshot(1));await blocked;});
 await assert.rejects(b.run(()=>{ran=true;}),SaveConflict);assert.equal(ran,false);
 release();await first;
});
test('no-op restoration does not rewrite records or increment their revision',async()=>{
 const env=environment();const a=make(env);a.read();await a.run(()=>a.save(snapshot()));
 const before=env.storage.getItem(SAVE_KEY);const b=make(env);b.read();await b.run(()=>b.save(snapshot()));
 assert.equal(env.storage.getItem(SAVE_KEY),before);
});
test('legacy saves migrate without deleting the only old copy',async()=>{
 const env=environment();const legacy=JSON.stringify(snapshot());env.storage.setItem(LEGACY_SAVE_KEY,legacy);
 const a=make(env);assert.equal(a.read(),legacy);await a.run(()=>a.save(snapshot()));
 assert.equal(env.storage.getItem(LEGACY_SAVE_KEY),legacy);const fresh=env.storage.getItem(SAVE_KEY);
 env.storage.setItem(LEGACY_SAVE_KEY,'stale pre-upgrade pagehide');
 assert.equal(env.storage.getItem(SAVE_KEY),fresh);assert.equal(make(env).read(),fresh);
});
test('storage notifications invalidate a stale window before another move',async()=>{
 const env=environment();let conflicts=0;const a=make(env),b=make(env,{onConflict:()=>conflicts++});a.read();b.read();
 await a.run(()=>a.save(snapshot(1)));b.changed({key:SAVE_KEY,storageArea:env.storage});assert.equal(conflicts,1);
});
test('closed and unsupported windows cannot write shared saves',async()=>{
 const env=environment();let warnings=0;const a=new MatchStore(env.storage,null,{onWarning:()=>warnings++});a.read();await a.run(()=>a.save(snapshot()));
 assert.equal(env.storage.getItem(SAVE_KEY),null);assert.ok(warnings>0);
 const b=make(env);b.read();b.close();assert.equal(await b.run(()=>b.save(snapshot())),false);assert.equal(env.storage.getItem(SAVE_KEY),null);
});
test('reload reads the latest revision and permits continued play',async()=>{
 const env=environment();const a=make(env);a.read();await a.run(()=>a.save(snapshot(1)));
 const b=make(env);assert.equal(JSON.parse(b.read()).commands.length,1);await b.run(()=>b.save(snapshot(2)));
 const value=JSON.parse(env.storage.getItem(SAVE_KEY));assert.equal(value.commands.length,2);assert.equal(value._storage.revision,2);
});
test('storage becoming inaccessible mid-match degrades to unsaved play',async()=>{
 const env=environment();let warnings=0;const a=make(env,{onWarning:()=>warnings++});a.read();
 env.storage.getItem=()=>{throw new Error('denied');};let ran=false;
 await a.run(()=>{ran=true;a.save(snapshot(1));});assert.ok(ran);assert.ok(warnings>0);
});
test('an exposed but denied Locks API runs once without shared writes',async()=>{
 const env=environment();env.locks.request=async()=>{throw new Error('SecurityError');};
 const a=make(env);a.read();let runs=0;await a.run(()=>{runs++;a.save(snapshot());});
 assert.equal(runs,1);assert.equal(env.storage.getItem(SAVE_KEY),null);
});
test('storage events during engine loading are read at startup, not treated as errors',()=>{
 const env=environment();const a=make(env);env.storage.setItem(SAVE_KEY,JSON.stringify(snapshot(1)));
 assert.doesNotThrow(()=>a.changed({key:SAVE_KEY,storageArea:env.storage}));assert.equal(JSON.parse(a.read()).commands.length,1);
});

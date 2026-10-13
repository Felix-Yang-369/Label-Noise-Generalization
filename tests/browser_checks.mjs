// Run in cua_repl with the current demo tab; exercises the real browser DOM.
import assert from 'node:assert/strict';

export async function checkSelectionLockedDuringRequest(tab) {
  const cdp = await tab.capabilities.get('cdp');
  await cdp.send('Network.enable');
  await cdp.send('Network.emulateNetworkConditions', {
    offline:false, latency:2000, downloadThroughput:-1, uploadThroughput:-1,
  });
  try {
    await tab.playwright.getByRole('button', {name:'Predict & compare', exact:false}).click();
    await tab.getAXState({emit:false});
    assert.equal(await tab.playwright.getByRole('combobox', {name:'Model',exact:true}).isEnabled(),false,
      'Model selector must be locked while the current request is running; prevent stale results under a new selection.');
    assert.equal(await tab.playwright.locator('#field-Area').isEnabled(),false,
      'Input values must not change while an earlier prediction is running.');
    assert.equal(await tab.playwright.getByRole('button',{name:'Load example',exact:false}).isEnabled(),false);
  } finally {
    await tab.playwright.getByRole('button',{name:'Predict & compare',exact:false}).waitFor({state:'visible',timeoutMs:15000});
    await cdp.send('Network.emulateNetworkConditions', {
      offline:false, latency:0, downloadThroughput:-1, uploadThroughput:-1,
    });
  }
  return 'Slow-request interaction check passed';
}

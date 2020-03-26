clear all;
clc;
% function out = model
% radius 14.5mm

import com.comsol.model.*
import com.comsol.model.util.*

Input = [];
Data = [];

NN = 8; LL0 = 1.45/NN;
for rr1 = 1:1:NN
     LL1 = rr1*LL0;
for rr2 = 1:1:NN
     LL2 = rr2*LL0;
for rr3 = 1:1:NN
     LL3 = rr3*LL0;
for rr4 = 1:1:NN
     LL4 = rr4*LL0;
for rr5 = 1:1:NN
     LL5 = rr5*LL0;
     
     arg_input = [LL1,LL2,LL3,LL4,LL5];
     Input = [Input arg_input'];
tic

model = ModelUtil.create('Model');

model.modelPath('F:\02_research_work\003_future\AI');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);
model.component('comp1').geom('geom1').axisymmetric(true);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('ta', 'ThermoacousticsSinglePhysics', 'geom1');
model.component('comp1').physics.create('acpr', 'PressureAcoustics', 'geom1');

model.study.create('std1');
model.study('std1').setGenConv(true);
model.study('std1').create('freq', 'Frequency');
model.study('std1').feature('freq').activate('ta', true);
model.study('std1').feature('freq').activate('acpr', true);

model.component('comp1').geom('geom1').lengthUnit('cm');

model.param.set('ww', '5[mm]');
model.param.set('d', '20[mm]');
model.param.set('f0', '500[Hz]');
model.param.set('dvisc', '0.22[mm]*sqrt(100[Hz]/f0)');
model.param.set('f_max', '6400[Hz]');
model.param.set('lbd', '343[m/s]/f_max');

model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size',  [LL1,15]);
model.component('comp1').geom('geom1').run('r1');

model.component('comp1').geom('geom1').create('r2', 'Rectangle');
model.component('comp1').geom('geom1').feature('r2').set('size', [LL2,15]);
model.component('comp1').geom('geom1').feature('r2').set('pos', {'0' 'ww*3'});
model.component('comp1').geom('geom1').run('r2');

model.component('comp1').geom('geom1').create('r3', 'Rectangle');
model.component('comp1').geom('geom1').feature('r3').set('size', [LL3,15]);
model.component('comp1').geom('geom1').feature('r3').set('pos', {'0' 'ww*3*2'});
model.component('comp1').geom('geom1').run('r3');

model.component('comp1').geom('geom1').create('r4', 'Rectangle');
model.component('comp1').geom('geom1').feature('r4').set('size', [LL4,15]);
model.component('comp1').geom('geom1').feature('r4').set('pos', {'0' 'ww*3*3'});
model.component('comp1').geom('geom1').run('r4');

model.component('comp1').geom('geom1').create('r5', 'Rectangle');
model.component('comp1').geom('geom1').feature('r5').set('size', [LL5,15]);
model.component('comp1').geom('geom1').feature('r5').set('pos', {'0' 'ww*3*4'});
model.component('comp1').geom('geom1').run('r5');

model.component('comp1').geom('geom1').create('r6', 'Rectangle');
model.component('comp1').geom('geom1').feature('r6').set('size', [1.45 1.2]);
model.component('comp1').geom('geom1').feature('r6').set('pos', {'0' 'ww*3*5'});
model.component('comp1').geom('geom1').feature('r6').setIndex('layer', 0.2, 0);
model.component('comp1').geom('geom1').feature('r6').set('layertop', true);
model.component('comp1').geom('geom1').feature('r6').set('layerbottom', false);
model.component('comp1').geom('geom1').run('r6');
model.component('comp1').geom('geom1').create('r7', 'Rectangle');
model.component('comp1').geom('geom1').feature('r7').set('size', [1.45 1.2]);
model.component('comp1').geom('geom1').feature('r7').set('pos', [0 -1.2]);
model.component('comp1').geom('geom1').feature('r7').setIndex('layer', 0.2, 0);
model.component('comp1').geom('geom1').runPre('fin');
model.component('comp1').geom('geom1').run('r7');
model.component('comp1').geom('geom1').create('boxsel1', 'BoxSelection');
model.component('comp1').geom('geom1').run('boxsel1');
model.component('comp1').geom('geom1').create('boxsel2', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel2').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel2').set('entitydim', 1);
model.component('comp1').geom('geom1').feature('boxsel2').set('ymin', 8.65);
model.component('comp1').geom('geom1').run('boxsel2');
model.component('comp1').geom('geom1').run('boxsel2');
model.component('comp1').geom('geom1').create('boxsel3', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel3').set('entitydim', 1);
model.component('comp1').geom('geom1').feature('boxsel3').set('ymax', -1.1);
model.component('comp1').geom('geom1').feature('boxsel3').set('condition', 'allvertices');
model.component('comp1').geom('geom1').run('boxsel3');
model.component('comp1').geom('geom1').run('boxsel3');
model.component('comp1').geom('geom1').create('boxsel4', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel4').set('entitydim', 1);
model.component('comp1').geom('geom1').feature('boxsel4').set('xmin', 0.1);
model.component('comp1').geom('geom1').feature('boxsel4').set('condition', 'allvertices');
model.component('comp1').geom('geom1').run('boxsel4');
model.component('comp1').geom('geom1').run;

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup('def').func.create('eta', 'Piecewise');
model.component('comp1').material('mat1').propertyGroup('def').func.create('Cp', 'Piecewise');
model.component('comp1').material('mat1').propertyGroup('def').func.create('rho', 'Analytic');
model.component('comp1').material('mat1').propertyGroup('def').func.create('k', 'Piecewise');
model.component('comp1').material('mat1').propertyGroup('def').func.create('cs', 'Analytic');
model.component('comp1').material('mat1').propertyGroup('def').func.create('an1', 'Analytic');
model.component('comp1').material('mat1').propertyGroup('def').func.create('an2', 'Analytic');
model.component('comp1').material('mat1').propertyGroup.create('RefractiveIndex', 'Refractive index');
model.component('comp1').material('mat1').propertyGroup.create('NonlinearModel', 'Nonlinear model');
model.component('comp1').material('mat1').label('Air');
model.component('comp1').material('mat1').set('family', 'air');
model.component('comp1').material('mat1').propertyGroup('def').func('eta').set('arg', 'T');
model.component('comp1').material('mat1').propertyGroup('def').func('eta').set('pieces', {'200.0' '1600.0' '-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4'});
model.component('comp1').material('mat1').propertyGroup('def').func('eta').set('argunit', 'K');
model.component('comp1').material('mat1').propertyGroup('def').func('eta').set('fununit', 'Pa*s');
model.component('comp1').material('mat1').propertyGroup('def').func('Cp').set('arg', 'T');
model.component('comp1').material('mat1').propertyGroup('def').func('Cp').set('pieces', {'200.0' '1600.0' '1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4'});
model.component('comp1').material('mat1').propertyGroup('def').func('Cp').set('argunit', 'K');
model.component('comp1').material('mat1').propertyGroup('def').func('Cp').set('fununit', 'J/(kg*K)');
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('expr', 'pA*0.02897/R_const[K*mol/J]/T');
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('args', {'pA' 'T'});
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('dermethod', 'manual');
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('argders', {'pA' 'd(pA*0.02897/R_const/T,pA)'; 'T' 'd(pA*0.02897/R_const/T,T)'});
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('argunit', 'Pa,K');
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('fununit', 'kg/m^3');
model.component('comp1').material('mat1').propertyGroup('def').func('rho').set('plotargs', {'pA' '0' '1'; 'T' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').func('k').set('arg', 'T');
model.component('comp1').material('mat1').propertyGroup('def').func('k').set('pieces', {'200.0' '1600.0' '-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4'});
model.component('comp1').material('mat1').propertyGroup('def').func('k').set('argunit', 'K');
model.component('comp1').material('mat1').propertyGroup('def').func('k').set('fununit', 'W/(m*K)');
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('expr', 'sqrt(1.4*R_const[K*mol/J]/0.02897*T)');
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('args', {'T'});
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('dermethod', 'manual');
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('argunit', 'K');
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('fununit', 'm/s');
model.component('comp1').material('mat1').propertyGroup('def').func('cs').set('plotargs', {'T' '273.15' '373.15'});
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('funcname', 'alpha_p');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('expr', '-1/rho(pA,T)*d(rho(pA,T),T)');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('args', {'pA' 'T'});
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('argunit', 'Pa,K');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('fununit', '1/K');
model.component('comp1').material('mat1').propertyGroup('def').func('an1').set('plotargs', {'pA' '101325' '101325'; 'T' '273.15' '373.15'});
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('funcname', 'muB');
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('expr', '0.6*eta(T)');
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('args', {'T'});
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('argunit', 'K');
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('fununit', 'Pa*s');
model.component('comp1').material('mat1').propertyGroup('def').func('an2').set('plotargs', {'T' '200' '1600'});
model.component('comp1').material('mat1').propertyGroup('def').set('thermalexpansioncoefficient', '');
model.component('comp1').material('mat1').propertyGroup('def').set('molarmass', '');
model.component('comp1').material('mat1').propertyGroup('def').set('bulkviscosity', '');
model.component('comp1').material('mat1').propertyGroup('def').set('relpermeability', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('def').set('dynamicviscosity', 'eta(T)');
model.component('comp1').material('mat1').propertyGroup('def').set('ratioofspecificheat', '1.4');
model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'0[S/m]' '0' '0' '0' '0[S/m]' '0' '0' '0' '0[S/m]'});
model.component('comp1').material('mat1').propertyGroup('def').set('heatcapacity', 'Cp(T)');
model.component('comp1').material('mat1').propertyGroup('def').set('density', 'rho(pA,T)');
model.component('comp1').material('mat1').propertyGroup('def').set('thermalconductivity', {'k(T)' '0' '0' '0' 'k(T)' '0' '0' '0' 'k(T)'});
model.component('comp1').material('mat1').propertyGroup('def').set('soundspeed', 'cs(T)');
model.component('comp1').material('mat1').propertyGroup('def').set('thermalexpansioncoefficient', {'alpha_p(pA,T)' '0' '0' '0' 'alpha_p(pA,T)' '0' '0' '0' 'alpha_p(pA,T)'});
model.component('comp1').material('mat1').propertyGroup('def').set('molarmass', '0.02897');
model.component('comp1').material('mat1').propertyGroup('def').set('bulkviscosity', 'muB(T)');
model.component('comp1').material('mat1').propertyGroup('def').addInput('temperature');
model.component('comp1').material('mat1').propertyGroup('def').addInput('pressure');
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').set('n', '');
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').set('ki', '');
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').set('n', {'1' '0' '0' '0' '1' '0' '0' '0' '1'});
model.component('comp1').material('mat1').propertyGroup('RefractiveIndex').set('ki', {'0' '0' '0' '0' '0' '0' '0' '0' '0'});
model.component('comp1').material('mat1').propertyGroup('NonlinearModel').set('BA', '(def.gamma+1)/2');
model.component('comp1').material('mat1').materialType('nonSolid');
model.component('comp1').material('mat1').set('family', 'air');

model.component('comp1').physics('ta').selection.named('geom1_boxsel1');
model.component('comp1').physics('acpr').selection.named('geom1_boxsel1');
model.component('comp1').physics('acpr').create('pwr1', 'PlaneWaveRadiation', 1);
model.component('comp1').physics('acpr').feature('pwr1').selection.named('geom1_boxsel2');
model.component('comp1').physics('acpr').create('pwr2', 'PlaneWaveRadiation', 1);
model.component('comp1').physics('acpr').feature('pwr2').selection.named('geom1_boxsel3');
model.component('comp1').physics('acpr').feature('pwr2').create('ipf1', 'IncidentPressureField', 1);
model.component('comp1').physics('acpr').feature('pwr2').feature('ipf1').selection.named('geom1_boxsel3');
model.component('comp1').physics('acpr').feature('pwr2').feature('ipf1').set('pamp', 1);
model.component('comp1').physics('acpr').feature('pwr2').feature('ipf1').set('c', 343);

model.component('comp1').multiphysics.create('atb1', 'AcousticThermoacousticBoundary', 1);
model.component('comp1').multiphysics('atb1').selection.all;

model.component('comp1').mesh('mesh1').create('bl1', 'BndLayer');
model.component('comp1').mesh('mesh1').feature('bl1').create('blp', 'BndLayerProp');
model.component('comp1').mesh('mesh1').feature('bl1').feature('blp').set('inittype', 'blhmin');
model.component('comp1').mesh('mesh1').feature('bl1').feature('blp').selection.named('geom1_boxsel4');
model.component('comp1').mesh('mesh1').feature('bl1').feature('blp').set('blhmin', 'dvisc/5');
model.component('comp1').mesh('mesh1').feature('size').set('custom', true);
model.component('comp1').mesh('mesh1').feature('size').set('hmax', 'lbd/5');
model.component('comp1').mesh('mesh1').feature('size').set('hmin', 'lbd/6');
model.component('comp1').mesh('mesh1').run;

model.study('std1').feature('freq').set('plist', 'range(20,20,5000)');
model.study('std1').run;

[Average, unit2] = mphmean(model,'abs(acpr.p_t)^2','line','selection','geom1_boxsel2');
Data=[Data,Average'];

m = size(Data)

save('AI_try.mat','Input','Data');

toc

end
end
end
end
end



OPENQASM 2.0;
include "qelib1.inc";
qreg q179[3];
rx(pi/2) q179[0];
cx q179[1],q179[2];
rx(3*pi/2) q179[1];
cx q179[0],q179[1];
rx(pi/4) q179[1];

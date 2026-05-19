OPENQASM 2.0;
include "qelib1.inc";
qreg q995[3];
cx q995[0],q995[1];
rx(3*pi/4) q995[1];
rx(pi/2) q995[0];
cx q995[0],q995[1];
cx q995[2],q995[1];

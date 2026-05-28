OPENQASM 2.0;
include "qelib1.inc";
qreg q724[3];
cx q724[0],q724[1];
cx q724[1],q724[2];
cx q724[1],q724[0];
cx q724[0],q724[1];
rx(pi/4) q724[0];

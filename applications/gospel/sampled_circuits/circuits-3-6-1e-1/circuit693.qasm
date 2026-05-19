OPENQASM 2.0;
include "qelib1.inc";
qreg q694[3];
rx(3*pi/4) q694[2];
cx q694[1],q694[2];
cx q694[0],q694[1];
rx(pi/4) q694[1];

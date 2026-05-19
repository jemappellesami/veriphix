OPENQASM 2.0;
include "qelib1.inc";
qreg q158[4];
rx(3*pi/4) q158[3];
cx q158[2],q158[3];
cx q158[1],q158[2];
cx q158[0],q158[1];
rx(pi/4) q158[1];

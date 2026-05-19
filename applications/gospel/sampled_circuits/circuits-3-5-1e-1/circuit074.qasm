OPENQASM 2.0;
include "qelib1.inc";
qreg q75[3];
rx(pi/2) q75[1];
rx(7*pi/4) q75[2];
cx q75[1],q75[2];
cx q75[1],q75[0];

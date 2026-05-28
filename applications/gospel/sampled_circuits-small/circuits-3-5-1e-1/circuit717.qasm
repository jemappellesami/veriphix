OPENQASM 2.0;
include "qelib1.inc";
qreg q718[3];
rx(7*pi/4) q718[0];
cx q718[1],q718[0];
cx q718[2],q718[1];
cx q718[0],q718[1];

OPENQASM 2.0;
include "qelib1.inc";
qreg q185[3];
rx(3*pi/2) q185[2];
cx q185[1],q185[2];
cx q185[0],q185[1];

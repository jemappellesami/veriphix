OPENQASM 2.0;
include "qelib1.inc";
qreg q185[5];
cx q185[3],q185[4];
cx q185[3],q185[2];
cx q185[2],q185[1];
cx q185[0],q185[1];
rx(pi/4) q185[1];

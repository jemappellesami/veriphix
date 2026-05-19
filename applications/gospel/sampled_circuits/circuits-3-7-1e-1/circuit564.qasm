OPENQASM 2.0;
include "qelib1.inc";
qreg q565[3];
rx(pi/2) q565[0];
cx q565[1],q565[2];
rz(pi/4) q565[2];
cx q565[2],q565[1];
cx q565[1],q565[0];
